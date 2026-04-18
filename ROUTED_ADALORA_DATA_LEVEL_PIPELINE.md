# Giải thích chi tiết luồng học mức Dataset (Dataset-level) cho Routed AdaLoRA-L

Dưới đây là tài liệu giải thích cơ chế hoạt động của thuật toán **Routed AdaLoRA-L** tại mức học toàn tập dữ liệu (dataset-level fine-tuning). 

Tài liệu này bao quát 4 phần chính:
1. Cách tính điểm và phân tách Layer Chung - Riêng.
2. Cơ chế đóng băng (Freeze) Layer.
3. Quá trình Huấn luyện (Training).
4. Quá trình Đánh giá (Testing).

---

## 1. Tính toán điểm Layer Chung và Riêng (Layer Importance & Allocation)

Mục tiêu của chặng này là tìm ra:
- **`common_layers`** (Layer dùng chung): Những layer bị ảnh hưởng nhiều nhất bởi đại đa số các APIs.
- **`api_specific_layers`** (Layer đặc thù): Những layer có điểm ảnh hưởng mạnh cho một API cụ thể nhưng KHÔNG nằm trong tập layer chung.

**Cách thực hiện (tại `knowledge-locator/`):**

*   **Tính điểm ảnh hưởng (Importance Score):** Trong script `api_importance_calculate.py`, mô hình xử lý tính tiến trình theo từng API trong tập dữ liệu. Dựa trên probing inputs (gradient/activation), một lớp `APILayerImportance` sẽ tính độ quan trọng (importance score) của từng layer cho từng API riêng lẻ. Kết quả này sẽ được trung bình cộng trên tất cả các test case thuộc chung 1 API, tạo ra bảng xếp hạng layer cho API đó.
*   **Xác định `common_layers` (Layer chung):** Tại `locate.py`, code sẽ thống kê tần suất xuất hiện của các layer nằm trong nửa top-đầu bảng xếp hạng của tất cả các APIs. N layer có tần suất xuất hiện nhiều nhất (ví dụ: `8 layer` đầu đối với DeepSeek `REMOVE_LAYER = [8, ...]`) sẽ được gom vào danh sách `common_layers`.
*   **Xác định `api_specific_layers` (Layer riêng):** Với mẫu layer của từng API, code duyệt từ top ảnh hưởng mạnh nhất xuống. Nếu layer có mặt trong `common_layers`, nó sẽ bị loại đi (bỏ qua). Chỉ giữ lại đúng $M$ layer (VD: `10 layer`) không giao với lớp `common_layers` làm lớp chuyên biệt của API đó. 
*   **Xuất Config:** Hai tập layer này được xuất ra file `routed_layer_config.json`.

---

## 2. Cơ chế Đóng băng (Freeze)

Dựa theo cấu hình layer đã tính ở trên, Routed AdaLoRA xử lý đóng băng mạng theo cơ chế Routing Adapter. Cụ thể:

*   **Toàn bộ base model:** Đều bị **đóng băng (freeze)** giống chuẩn LoRA/AdaLoRA. Không có trọng số chính (pre-trained weights) nào được học hay thay đổi.
*   **`common_layers` (Lớp phổ biến chung):** Các layer này không sinh thêm bất kỳ tham số PEFT/Adapter nào. Về bản chất, chúng được **hoàn toàn đóng băng** và đi thẳng qua mạng Feed-forward/Attention ban đầu. Đây là cách giúp mô hình giữ lại "độ khái quát" và không để các API phá hủy kiến thức ngôn ngữ chung nền tảng.
*   **`api_specific_layers`:** Chỉ có các layer này mới được gắn thêm tham số huấn luyện hạ dòng (trainable params - Low Rank Adapter).

---

## 3. Lúc Train diễn ra như thế nào?

Quá trình học diễn ra tại `codellmeditor/models/routed_adalora/routed_adalora_main.py` (hàm `execute_routed_adalora`).

1.  **Nhóm dữ liệu (Group Requests by API):** Toàn bộ dataset được gom nhóm dựa trên key `target_api`.
2.  **Khởi tạo Adapter chuyên biệt:**
    *   Do thư viện PEFT mặc định của AdaLoRA chỉ hỗ trợ active 1 adapter tại một thời điểm trong qua trình khởi tạo, module tự động fallback sang cơ chế `LoRAConfig` nhằm xử lý multi-adapter.
    *   Với mỗi API $A_k$, một Adapter phân cực $Adpt_k$ được sinh ra. Adapter này chỉ gắn trên vị trí của `api_specific_layers` của chính nó. Không ảnh hưởng đến các layer khác. Mô hình base sẽ chứa nhiều cụm Adapter độc lập.
3.  **Vòng lặp Training (Round-robin fashion):**
    *   Code xáo trộn (shuffle) lại thứ tự các API qua từng epoch.
    *   **Routing (Định tuyến):** Khi chuẩn bị train một lô dữ liệu (batch) thuộc $API_k$, mô hình sẽ gọi `model.set_adapter(Adapter_k)` để kích hoạt các trọng số chuyên trách. Tất cả tham số của ngách adapter khác được đưa vào trạng thái đóng băng tạm thời/inactive.
    *   **Forward & Backward:** Forward input của $API_k$ qua mô hình, đi qua `common_layers` bằng bộ trọng số nguyên thủy, đi qua `api_specific_layers` với sự hỗ trợ của LORA Adapter $Adpt_k$. Tính chuẩn gradient, update riêng cho $Adpt_k$.

=> Kết quả là ta có một base-model cõng theo hàng trăm Adapter nhánh, mỗi nhánh chỉ được đụng vào các "nơ-ron" nhạy cảm với API mà nó phục vụ. Tránh việc triệt tiêu kiến thức chéo (catastrophic forgetting lẫn nhau).

---

## 4. Quá trình Đánh giá (Test) - Testing Phase

Quá trình test (`editor.edit_dataset_level` và `compute_edit_quality`) tuân thủ tính chuyên hóa của tập train:

1.  **Iterate Over Pipeline:** Trình test đi qua từng case một lẻ tẻ trên tập evaluate.
2.  **Xác định API và Activate:** Lấy `target_api` từ mẫu eval. Tìm xem API này tên Adapter map là gì. 
3.  **Routing (Chuyển kênh):**
    *   Sử dụng lệnh `edited_model.set_adapter(adapter_map[api_name])` để chuyển trạng thái Base Model chĩa toàn bộ kênh tính toán sang Adapter được train chuyên biệt của API mình đang muốn sinh.
    *   Trường hợp test là API mới chưa có mặt trong quá trình train (unseen), mô hình giữ nguyên kênh trước đó (cảnh báo fallback).
4.  **Generation & Inference:** Đưa `prompt` tương ứng vào (kể cả với các prompt dành cho portability, efficacy, generalizaion,...). Sinh batch chuỗi kết quả trên 1 tập Adapter được chọn.
5.  **Calculate Metrics:** Truyền output sang cho hệ thống metric EM (Exact Match), ROUGE, API EM... được thiết lập tự động trong `edapi_evaluate.py`. Tính toán ghi nhận bộ nhớ (Peak Memory) và Thời gian trôi qua, lưu vào tệp JSON.
