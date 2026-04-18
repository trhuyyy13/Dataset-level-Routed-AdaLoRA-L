# Giải thích Chuyên sâu Cấu trúc lõi: Dataset-level Routed AdaLoRA-L

Tài liệu này phân tích chi tiết tận cùng cấp độ source code (toán học, xử lý tensor, kiến trúc mạng) đằng sau phương pháp **Dataset-level Routed AdaLoRA-L** hiện đang được implement trong không gian EDAPI-Bench.

---

## 1. Module `importance_calculate.py` (Toán học & Tính điểm Tensors)

Quá trình này trả lời câu hỏi: *Làm sao mô hình biết chính xác Layer nào trong hàng chục Layer khổng lồ chứa kiến thức về hàm `pandas.DataFrame.map`?*

### 1.1. Xác định Token mục tiêu (Token Alignment)
- Mã nguồn nhận vào `input_text` và `target_api`.
- Bộ Tokenizer chia nhỏ văn bản và sử dụng tham số `return_offsets_mapping=True` (ánh xạ từng token ID ngược lại vị trí chuỗi ký tự gốc trong văn bản).
- Chạy thuật toán đối chiếu chuỗi (`api_start` lặp qua offsets) để nhặt ra **chính xác các vị trí token (`api_token_indices`)** tạo nên chuỗi `target_api`.
  
### 1.2. Tạo mặt nạ Focused Loss (Masked Cross-Entropy)
LLM là mô hình Next-token Prediction (dự đoán token kế tiếp), nên nhãn (`shift_labels`) lúc nào cũng trượt sang trái 1 nhịp (`input_ids[..., 1:]`).
- Lấy `logits` dự đoán $\rightarrow$ tính Cross-Entropy Loss TỪNG TOKEN `token_losses = loss_fct(...)` nhưng chưa cộng tổng.
- Tạo một biến Tensor `mask` mảng 0 toàn tập. Ta gán `mask[0, idx - 1] = 1.0` (phải trừ 1 vì label bị shift left).
- **Focused Loss**: 
  $$ \mathcal{L}_{foc} = \sum (\text{token\_losses} \odot \text{mask}) $$
  $\rightarrow$ Giá trị Loss này hiện chỉ hoàn toàn chịu trách nhiệm cho các token của API. Gọi hàm `focused_loss.backward()` sẽ truyền ngược Gradient sinh ra từ sự "nhấn mạnh" này xuống toàn bộ trọng số mạng.

### 1.3. Tính điểm Importance cho Module bằng Gradient
Hệ thống sẽ chạy qua các tham số edit mục tiêu (VD: `down_proj` của thành phần MLP đối với DeepSeek, hoặc `q_proj, v_proj` với StarCoder).
Hàm tính điểm `imp_func` được truyền vào:
1. **Fisher Information:** Trọng số đạo hàm bình phương trung bình.
   ```python
   def _compute_fisher_information(param, grad):
       return (grad ** 2).mean()
   ```
2. **Sensitivity:** Biên độ thay đổi tuyệt đối.
   ```python
   def _compute_sensitivity(param, grad):
       return (param * grad).abs().mean()
   ```
Điểm của từng module được tổng hợp. Tổng điểm module cùng thuộc Layer $L$ sẽ được cộng dồn lại thành `layer_scores`. Ta thu được Ranking Layer Quan trọng cho từng mẫu code.

---

## 2. Module `locate.py` (Cơ chế phân tách Common vs. Specific Layers)

Đầu vào là `importance.json` chứa Ranking layer được trung bình từ tất cả các samples của mỗi API. Bài toán đặt ra là phải đóng băng các vùng kiến thức chung ngôn ngữ (Python grammar).

1. Lấy **50% Layers đứng top đầu** (Nửa trên của Rank) của MỖI API.
2. Ném tất cả các layer này vào chung một "bể đếm" (frequency counter) để xem Layer nào được nhiều API khác nhau "vote" là quan trọng nhất.
3. Cắt lấy $K$ Layer có tần suất xuất hiện cao nhất (`REMOVE_LAYER = 8` đối với deepseek-1.3b).
   - Đây chính là **`common_layers`**. Mặc định những khu vực này là lõi cấu trúc lập trình chung chứ không đại diện cho 1 lệnh API cụ thể nào.
   - Sẽ bị **đóng băng (frozen)** vĩnh viễn trong quá trình update.
4. Quành lại từng API: Lọc bộ điểm gốc từ trên xuống dưới, gõ bỏ các `common_layers`, lấy $M$ layer quan trọng kế tiếp (`LAYER_NUM = 8`).
   - Mảng giữ lại là **`api_specific_layers`** $\rightarrow$ Lưu xuống `routed_layer_config.json`.

---

## 3. Module `routed_adalora_main.py` (Lõi Định tuyến Bằng PEFT)

Đây là nơi biến ý tưởng multi-APIs thành hiện thực.

### 3.1 Khởi tạo Mạng Lưới Adapter
Hệ thống kiểm tra thư viện HuggingFace `PEFT`. Vì biến thể AdaLoRA giới hạn kĩ thuật RankAllocator chỉ cho 1 mảng adapter thao tác tại một thời điểm (khi `inference_mode=False`), hệ thống sẽ Fallback khởi tạo bằng cấu trúc **LoraConfig** để hỗ trợ Multi-Trainable Adapters.

- Chạy vòng lặp cấp phát cho từng cụm rẽ nhánh (API):
  ```python
  model.add_adapter(
      adapter_name="pandas_DataFrame_map",
      peft_config=LoraConfig(
          layers_to_transform=[3, 7, 10, ...], # Chỉ apply lên Specific Layers
          target_modules=["down_proj"],
          r=8
      )
  )
  ```
- Kết quả: VRAM tiêu thụ rất í. 1.35 Tỷ param của Base Model đứng im dể làm kênh trung chuyển (backbone). Hệ thống mọc ra hàng chục cụm Neural con (rank=8), tính tổng Trainable Params chỉ vỏn vẹn `724,992 (0.05%)`.

### 3.2 Huấn luyện theo Block Route (Round-Robin)
- Data Dataloader thay vì trộn tung toé toàn dataset, nó gom data theo từng khối ứng với API riêng.
- Trong `Epoch N`: Randomize danh sách tên API.
  - Đi đến khối dữ liệu của bộ API `X`.
  - Thay vì backward tất cả, bộ chọn đường (Router) chuyển mạng nơ-ron trọng tâm qua `X`: 
    `model.set_adapter(api_name_X)`
  - Loss tính tới đâu, Optimizer step cập nhật thẳng về cụm rẽ nhánh (Adapter) của API `X`. Các nhánh API `Y, Z` ngủ đông, hoàn toàn an toàn $\rightarrow$ Triệt tiêu 100% Catastrophic Forgetting chéo.

---

## 4. Module `editor.py` & `edapi_evaluate.py` (Đánh giá sau khi đào tạo)

Phương pháp này chỉ nạp Model Weights lên VRAM Memory **ĐÚNG 1 LẦN** (Khác với phương pháp cũ làm thao tác Load Weights từ ổ cứng HDD/SSD lên cho từng API, mất hàng tiếng đồng hồ).

Sau model training phase, pipeline tự động thả trôi qua Evaluation phase:
- Mở danh sách bài test.
- Đọc bài test mẫu $T$. Biến số `target_api` trong bài Test chỉ ra tên của hàm cũ (Deprecated).
- Pipeline bật công tắc não: `model.set_adapter(test.target_api)`.
- Nhờ công tắc này, Token luân chuyển đi qua Base Model $\rightarrow$ vào đúng specific layers $\rightarrow$ đập vào LoRA của $T$ $\rightarrow$ Sinh ra biến dạng Token sửa thành Tân API mã nguồn (Updated API).
- Thang đo chấm dựa trên 4 metric:
  1. `Efficacy (E)`: Exact match - Kiểm tra chuỗi Model Gen sinh ra có y chang Chuỗi Code Updated hay không. Lấy mẫu từ Prompt gốc huấn luyện.
  2. `Generalization (G)`: Y hệt bước trên, nhưng Input Text đầu vào là Prompt đã được viết lại bằng ngữ pháp tiếng Anh khác đi `rephrase_prompt`.
  3. `Portability (P)`: Cho Model Gen một hàm Code ứng dụng liên thông logic hoặc dùng thư viện lồng sâu hơn với `portability_prompt`, xem model có nhớ cách gõ lệnh mới không.
  4. `Specificity (S)`: Cực kì quan trọng! Prompt hỏi về những lệnh/API **Xung quanh/Hàng Xóm** nhưng tuyệt đối không được Edit chúng. Trạng thái Local Adapter sẽ giúp chứng minh rằng Update Target API không làm nhiễu Neighbor APIs.
