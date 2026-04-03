from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: FunctionDefinition


class ToolChoiceFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)


class ToolChoiceObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: ToolChoiceFunction


class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    arguments: str


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: Literal["function"]
    function: ToolCallFunction


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

    @model_validator(mode="after")
    def validate_tool_fields(self) -> "ChatMessage":
        if self.role in {"system", "user"} and self.content is None:
            raise ValueError("content is required for system and user messages")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")
        if self.role != "assistant" and self.tool_calls is not None:
            raise ValueError("tool_calls are only valid for assistant messages")
        return self


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: List[ChatMessage]
    stream: bool = False
    request_id: Optional[str] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[Literal["auto", "none"], ToolChoiceObject]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

    @model_validator(mode="after")
    def validate_messages(self) -> "ChatCompletionRequest":
        if not self.messages:
            raise ValueError("messages must not be empty")
        if self.tool_choice is not None and not self.tools:
            raise ValueError("tool_choice requires tools")
        return self


class ChatCompletionError(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ChatCompletionErrorResponse(BaseModel):
    error: ChatCompletionError
