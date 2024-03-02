import pdb
from copy import deepcopy
from typing import Any, Dict, List, Optional

import jsonref
import requests
import yaml

from functionary.openai_types import Function


def convert_data_type(param_type: str) -> str:
    """convert data_type to typescript data type

    Args:
        param_type (str): param_type

    Returns:
        str: param type in typescript
    """
    if param_type == "integer" or param_type == "float":
        return "number"
    return param_type


def get_param_type(param: Dict) -> str:
    """get param_type of parameter

    Args:
        param (Dict): param dict in properties

    Returns:
        str: _description_
    """
    param_type = "any"
    if "type" in param:
        raw_param_type = param["type"]
        if type(raw_param_type) is list:
            param_type = " | ".join(raw_param_type)
        else:
            param_type = raw_param_type

    else:  # in many cases, the json schema contains: oneOf instead of "type"
        if "oneOf" in param:
            one_of_types = []
            for item in param["oneOf"]:
                if "type" in item:
                    one_of_types.append(convert_data_type(item["type"]))
            one_of_types = list(set(one_of_types))
            param_type = " | ".join(one_of_types)
    return convert_data_type(param_type)


def get_format_param(param: Dict) -> Optional[str]:
    """Get "format" from param. There are cases where format is not directly in param but in oneOf

    Args:
        param (Dict): _description_

    Returns:
        Optional[str]: _description_
    """
    if "format" in param:
        return param["format"]
    if "oneOf" in param:
        formats = []
        for item in param["oneOf"]:
            if "format" in item:
                formats.append(item["format"])
        if len(formats) > 0:
            return " or ".join(formats)
    return None


def get_param_info(param: Dict) -> Optional[str]:
    """get additional information about parameter such as: format, default value, min, max, ...

    Args:
        param (Dict): _description_

    Returns:
        Optional[str]: _description_
    """
    param_type = param.get("type", "any")
    info_list = []
    if "description" in param:
        desc = param["description"]
        if not desc.endswith("."):
            desc += "."
        info_list.append(desc)

    if "default" in param:
        default_value = param["default"]
        if param_type == "string":
            default_value = f'"{default_value}"'  # if string --> add ""
        info_list.append(f"Default={default_value}.")

    format_param = get_format_param(param)
    if format_param is not None:
        info_list.append("Format=" + format_param)

    for field, field_name in [
        ("maximum", "Maximum"),
        ("minimum", "Minimum"),
        ("maxLength", "Maximum length"),
        ("minLength", "Minimum length"),
    ]:
        if field in param:
            info_list.append(f"{field_name}=" + str(param[field]))

    if len(info_list) > 0:
        result = "// " + " ".join(info_list)
        result = result.replace("\n", " ")
        return result
    return None


def append_new_param_info(
    info_list: List[str],
    param_declaration: str,
    comment_info: Optional[str],
    depth: int,
):
    """Append a new parameter with comment to the info_list

    Args:
        info_lines (List[str]): current info_list
        param_declaration (str): param: type
        comment_info (Optional[str]): information of comment
        depth (int): level of nested param
    """
    offset = ""
    if depth >= 1:
        offset = "".join(["    " for _ in range(depth)])
    if comment_info is not None:
        # if depth == 0:  # format: //comment\nparam: type
        info_list.append(f"{offset}{comment_info}")
        info_list.append(f"{offset}{param_declaration}")
    # else:  # format: param: type  // comment
    #     info_list.append(f"{offset}{param_declaration}    {comment_info}")
    else:
        info_list.append(f"{offset}{param_declaration}")


def get_enum_option_str(enum_options: List) -> str:
    """get enum option separated by: "|"

    Args:
        enum_options (List): list of options

    Returns:
        _type_: concatenation of options separated by "|"
    """
    # if each option is string --> add quote
    return " | ".join([f'"{v}"' if type(v) is str else str(v) for v in enum_options])


def get_array_typescript(
    param_name: Optional[str], param_dic: dict, depth: int = 0
) -> str:
    """recursive implementation for generating type script of array

    Args:
        param_name (Optional[str]): name of param, optional
        param_dic (dict): param_dic
        depth (int, optional): nested level. Defaults to 0.

    Returns:
        _type_: typescript of array
    """
    offset = ""
    if depth >= 1:
        offset = "".join(["    " for _ in range(depth)])
    items_info = param_dic.get("items", {})

    if len(items_info) == 0:
        if param_name is not None:
            return f"{offset}{param_name}: []"
        else:
            return "[]"
    array_type = get_param_type(items_info)
    if array_type == "object":
        info_lines = []
        child_lines = get_parameter_typescript(
            items_info.get("properties", {}), items_info.get("required", []), depth + 1
        )
        # if comment_info is not None:
        #    info_lines.append(f"{offset}{comment_info}")
        if param_name is not None:
            info_lines.append(f"{offset}{param_name}" + ": {")
        else:
            info_lines.append(f"{offset}" + "{")
        info_lines.extend(child_lines)
        info_lines.append(f"{offset}" + "}[]")
        return "\n".join(info_lines)

    elif array_type == "array":
        item_info = get_array_typescript(None, items_info, depth + 1)
        if param_name is None:
            return f"{item_info}[]"
        return f"{offset}{param_name}: {item_info.strip()}[]"

    else:
        if "enum" in items_info:
            item_type = get_enum_option_str(items_info["enum"])
            if param_name is None:
                return f"({item_type})[]"
            else:
                return f"{offset}{param_name}: ({item_type})[]"
        else:
            if param_name is None:
                return f"{array_type}[]"
            else:
                return f"{offset}{param_name}: {array_type}[],"


def get_parameter_typescript(properties, required_params, depth=0) -> List[str]:
    """Recursion, returning the information about parameters including data type, description and other information
    These kinds of information will be put into the prompt

    Args:
        properties (_type_): properties in parameters
        required_params (_type_): List of required parameters
        depth (int, optional): the depth of params (nested level). Defaults to 0.

    Returns:
        _type_: list of lines containing information about all parameters
    """
    tp_lines = []
    for param_name, param in properties.items():
        # Sometimes properties have "required" field as a list of string.
        # Even though its supposed to be not under properties. So we skip it
        if not isinstance(param, dict):
            continue
        # Param Description
        comment_info = get_param_info(param)
        # Param Name declaration
        param_declaration = f"{param_name}"
        if isinstance(required_params, list):
            if param_name not in required_params:
                param_declaration += "?"
        param_type = get_param_type(param)

        offset = ""
        if depth >= 1:
            offset = "".join(["    " for _ in range(depth)])

        if param_type == "object":  # param_type is object
            child_lines = get_parameter_typescript(
                param.get("properties", {}), param.get("required", []), depth + 1
            )
            if comment_info is not None:
                tp_lines.append(f"{offset}{comment_info}")

            param_declaration += ": {"
            tp_lines.append(f"{offset}{param_declaration}")
            tp_lines.extend(child_lines)
            tp_lines.append(f"{offset}" + "},")

        elif param_type == "array":  # param_type is an array
            item_info = param.get("items", {})
            if "type" not in item_info:  # don't know type of array
                param_declaration += ": [],"
                append_new_param_info(tp_lines, param_declaration, comment_info, depth)
            else:
                array_declaration = get_array_typescript(
                    param_declaration, param, depth
                )
                if not array_declaration.endswith(","):
                    array_declaration += ","
                if comment_info is not None:
                    tp_lines.append(f"{offset}{comment_info}")
                tp_lines.append(array_declaration)
        else:
            if "enum" in param:
                param_type = get_enum_option_str(param["enum"])
                # param_type = " | ".join([f'"{v}"' for v in param["enum"]])
            param_declaration += f": {param_type},"
            append_new_param_info(tp_lines, param_declaration, comment_info, depth)

    return tp_lines


def generate_schema_from_functions(
    functions: List[Function], namespace="functions"
) -> str:
    """
    Convert functions schema to a schema that language models can understand.
    """

    schema = "// Supported function definitions that should be called when necessary.\n"
    schema += f"namespace {namespace} {{\n\n"

    for function in functions:
        # Convert a Function object to dict, if necessary
        if not isinstance(function, dict):
            function = function.model_dump()
        function_name = function.get("name", None)
        if function_name is None:
            continue

        description = function.get("description", "")
        schema += f"// {description}\n"
        schema += f"type {function_name}"

        parameters = function.get("parameters", None)
        if parameters is not None and parameters.get("properties") is not None:
            parameters = deepcopy(jsonref.JsonRef.replace_refs(parameters))
            schema += " = (_: {\n"
            required_params = parameters.get("required", [])
            tp_lines = get_parameter_typescript(
                parameters.get("properties"), required_params, 0
            )
            schema += "\n".join(tp_lines)
            schema += "\n}) => any;\n\n"
        else:
            # Doesn't have any parameters
            schema += " = () => any;\n\n"

    schema += f"}} // namespace {namespace}"

    return schema


def generate_schema_from_openapi(
    specification: Dict[str, Any], description: str, namespace: str
) -> str:
    """
    Convert OpenAPI specification object to a schema that language models can understand.

    Input:
    specification: can be obtained by json.loads of any OpanAPI json spec, or yaml.safe_load for yaml OpenAPI specs

    Example output:

    // General Description
    namespace functions {

    // Simple GET endpoint
    type getEndpoint = (_: {
    // This is a string parameter
    param_string: string,
    param_integer: number,
    param_boolean?: boolean,
    param_enum: "value1" | "value2" | "value3",
    }) => any;

    } // namespace functions
    """

    description_clean = description.replace("\n", "")

    schema = f"// {description_clean}\n"
    schema += f"namespace {namespace} {{\n\n"

    for path_name, paths in specification.get("paths", {}).items():
        for method_name, method_info in paths.items():
            operationId = method_info.get("operationId", None)
            if operationId is None:
                continue
            description = method_info.get("description", method_info.get("summary", ""))
            schema += f"// {description}\n"
            schema += f"type {operationId}"

            if ("requestBody" in method_info) or (
                method_info.get("parameters") is not None
            ):
                schema += f"  = (_: {{\n"
                # Body
                if "requestBody" in method_info:
                    try:
                        body_schema = (
                            method_info.get("requestBody", {})
                            .get("content", {})
                            .get("application/json", {})
                            .get("schema", {})
                        )
                    except AttributeError:
                        body_schema = {}
                    for param_name, param in body_schema.get("properties", {}).items():
                        # Param Description
                        description = param.get("description")
                        if description is not None:
                            schema += f"// {description}\n"

                        # Param Name
                        schema += f"{param_name}"
                        if (
                            (not param.get("required", False))
                            or (param.get("nullable", False))
                            or (param_name in body_schema.get("required", []))
                        ):
                            schema += "?"

                        # Param Type
                        param_type = param.get("type", "any")
                        if param_type == "integer":
                            param_type = "number"
                        if "enum" in param:
                            param_type = " | ".join([f'"{v}"' for v in param["enum"]])
                        schema += f": {param_type},\n"

                # URL
                for param in method_info.get("parameters", []):
                    # Param Description
                    if description := param.get("description"):
                        schema += f"// {description}\n"

                    # Param Name
                    schema += f"{param['name']}"
                    if (not param.get("required", False)) or (
                        param.get("nullable", False)
                    ):
                        schema += "?"
                    if param.get("schema") is None:
                        continue
                    # Param Type
                    param_type = param["schema"].get("type", "any")
                    if param_type == "integer":
                        param_type = "number"
                    if "enum" in param["schema"]:
                        param_type = " | ".join(
                            [f'"{v}"' for v in param["schema"]["enum"]]
                        )
                    schema += f": {param_type},\n"

                schema += f"}}) => any;\n\n"
            else:
                # Doesn't have any parameters
                schema += f" = () => any;\n\n"

    schema += f"}} // namespace {namespace}"

    return schema


def generate_specification_from_openapi_url(
    openapi_url: str, proxies: dict = None
) -> str:
    # Make Request
    headers = {"Accept": "application/x-yaml, text/yaml, text/x-yaml, application/json"}
    response = requests.get(
        openapi_url, verify=False, headers=headers, timeout=60, proxies=proxies
    )

    if response.status_code == 200:
        # Trust content-type first
        if response.headers.get("Content-Type") is not None:
            if "application/json" in response.headers.get("Content-Type"):
                specification = response.json()
            else:
                specification = yaml.safe_load(response.text)
        elif response.url.endswith(".json"):
            specification = response.json()
        else:
            specification = yaml.safe_load(response.text)
        # Resolve references
        specification = deepcopy(jsonref.JsonRef.replace_refs(specification))
        return specification
