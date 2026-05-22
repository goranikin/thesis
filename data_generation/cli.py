import argparse
import types
from typing import Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


def _field_cli_name(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def _unwrap_annotation(annotation):
    origin = get_origin(annotation)
    if origin is types.UnionType:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _add_argument(parser: argparse.ArgumentParser, name: str, field: FieldInfo) -> None:
    annotation, optional = _unwrap_annotation(field.annotation)
    origin = get_origin(annotation)

    kwargs: dict = {}
    if field.description is not None:
        kwargs["help"] = field.description

    if origin is Literal:
        choices = get_args(annotation)
        kwargs["type"] = type(choices[0])
        kwargs["choices"] = list(choices)
    elif annotation is bool:
        kwargs["action"] = "store_true"
    elif annotation in (int, float, str):
        kwargs["type"] = annotation
    else:
        kwargs["type"] = str

    if field.default is not PydanticUndefined:
        kwargs["default"] = field.default
    elif optional:
        kwargs["default"] = None

    parser.add_argument(_field_cli_name(name), **kwargs)


def build_parser(
    config_type: type[BaseModel],
    *,
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    for name, field in config_type.model_fields.items():
        _add_argument(parser, name, field)
    return parser


def parse_config[T: BaseModel](
    config_type: type[T],
    argv: list[str] | None = None,
) -> T:
    parser = build_parser(config_type)
    namespace = parser.parse_args(argv)
    return config_type.model_validate(vars(namespace))
