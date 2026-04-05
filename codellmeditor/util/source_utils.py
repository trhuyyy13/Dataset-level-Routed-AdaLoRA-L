import re
import ast

def extract_first_func_bak(code:str):
    indent = len(re.search("^\s*", code).group(0))
    lines = code.split("\n")
    comment = ""
    if lines[0].startswith(" " * indent + "'''"):
        while len(lines) > 0:
            line = lines.pop(0)
            comment += line + "\n"
            if line.strip().endswith("'''"):
                break
    elif lines[0].startswith(" " * indent + '"""'):
        while len(lines) > 0:
            line = lines.pop(0)
            comment += line + "\n"
            if line.strip().endswith('"""'):
                break
    while len(lines) > 0:
        if lines[0].startswith(" " * indent + "#"):
            line = lines.pop(0)
            comment += line + "\n"
        else:
            break
    func = "\n".join(lines)
    func = re.split(r"\n {0,%d}[^\s#]" % indent, func, flags=re.M|re.S)[0]
    func = comment + func
    return func


def extract_first_func(code:str):
    lines = code.split("\n")
    while len(lines) > 0 and not lines[0].lstrip().startswith("def "):
        lines.pop(0)
    if len(lines) == 0:
        return code
    indent = len(re.search("^\s*", lines[0]).group(0))
    func = "\n".join(lines)
    func = re.split(r"\n {0,%d}[^\s#]" % indent, func, flags=re.M|re.S)[0]
    return func

def clean_pred(pred:str):
    lines = [line for line in pred.split("\n") if not line.strip().startswith("#")]
    return "\n".join(lines)


def extract_first_statement(pred:str, remove_space=True):
    def unclosed(_stmt:str):
        if len(_stmt) == 0:
            return True
        if _stmt.count("(") > _stmt.count(")"):
            return True
        if _stmt.count("[") > _stmt.count("]"):
            return True
        if _stmt.count("{") > _stmt.count("}"):
            return True
        if _stmt.rstrip().endswith("\\"):
            return True
        return False
    
    def normalize(_line:str):
        _line = _line.split("#")[0]
        _line = _line.strip().rstrip(" \\")
        _line = re.sub(r"\s+", " ", _line)
        if remove_space:
            _line = re.sub(r"\s+", "", _line)
        return _line

    
    lines = pred.split("\n")
    stmt = normalize(lines.pop(0))
    
    while unclosed(stmt) and len(lines) > 0:
        stmt += normalize(lines.pop(0))
    return stmt


def extract_first_api(pred, ref_dict, alias_dict):
    pkg_as = dict()
    for alias, name in alias_dict.items():
        alias_parts = alias.split(".")
        name_parts = name.split(".")
        while len(alias_parts) > 0 and len(name_parts) > 0 and alias_parts[-1] == name_parts[-1]:
            alias_parts.pop()
            name_parts.pop()
        pkg_alias, pkg_name = ".".join(alias_parts), ".".join(name_parts)
        if pkg_alias != pkg_name:
            pkg_as[pkg_alias] = pkg_name

    for mobj in re.finditer(r"([\w\.]+)\s*\(", pred):
        api = mobj.group(1).strip()
        if api == "":
            continue
        parts = api.split('.')
        if len(parts) == 2 and parts[0] in ref_dict:
            api = f"{ref_dict[parts[0]]}.{parts[1]}"
        if api in alias_dict:
            api = alias_dict[api]
        else:
            for pkg_alias, pkg_name in pkg_as.items():
                if api.startswith(f"{pkg_alias}."):
                    api = api.replace(f"{pkg_alias}.", f"{pkg_name}.")
                    break
        return api
    return ""


def extract_apis_in_first_stmt(pred, ref_dict, alias_dict):
    stmt = extract_first_statement(pred, False)
    pkg_as = dict()
    for alias, name in alias_dict.items():
        alias_parts = alias.split(".")
        name_parts = name.split(".")
        while len(alias_parts) > 0 and len(name_parts) > 0 and alias_parts[-1] == name_parts[-1]:
            alias_parts.pop()
            name_parts.pop()
        pkg_alias, pkg_name = ".".join(alias_parts), ".".join(name_parts)
        if pkg_alias != pkg_name:
            pkg_as[pkg_alias] = pkg_name

    apis = set()
    for mobj in re.finditer(r"([\w\.]+)\s*\(", stmt):
        api = mobj.group(1).strip()
        if api == "":
            continue
        parts = api.split('.')
        if len(parts) == 2 and parts[0] in ref_dict:
            api = f"{ref_dict[parts[0]]}.{parts[1]}"
        if api in alias_dict:
            api = alias_dict[api]
        else:
            for pkg_alias, pkg_name in pkg_as.items():
                if api.startswith(f"{pkg_alias}."):
                    api = api.replace(f"{pkg_alias}.", f"{pkg_name}.")
                    break
        apis.add(api)
    return list(apis)


def index_of_api(pred:str, target_apis, ref_dict, alias_dict):
    pkg_as = dict()
    for alias, name in alias_dict.items():
        alias_parts = alias.split(".")
        name_parts = name.split(".")
        while len(alias_parts) > 0 and len(name_parts) > 0 and alias_parts[-1] == name_parts[-1]:
            alias_parts.pop()
            name_parts.pop()
        pkg_alias, pkg_name = ".".join(alias_parts), ".".join(name_parts)
        if pkg_alias != pkg_name:
            pkg_as[pkg_alias] = pkg_name


    for mobj in re.finditer(r"([\w\.]+)\s*\(", pred):
        api = mobj.group(1).strip()
        if api == "":
            continue
        parts = api.split('.')
        if len(parts) == 2 and parts[0] in ref_dict:
            api = f"{ref_dict[parts[0]]}.{parts[1]}"
        if api in alias_dict:
            api = alias_dict[api]
        else:
            for pkg_alias, pkg_name in pkg_as.items():
                if api.startswith(f"{pkg_alias}."):
                    api = api.replace(f"{pkg_alias}.", f"{pkg_name}.")
                    break
        if api in target_apis:
            idx = pred.index(mobj.group(0))
            return idx
    return 0


def normalize_stmt(stmt):
    mobj = re.match(r"(.*?)=(.*)$", stmt)
    if not mobj:
        return stmt
    left, right = mobj.group(1), mobj.group(2)
    if left.count("(") != left.count(")"):
        return stmt
    left = re.sub(r"\w+", "_", left)
    return f"{left}={right}"