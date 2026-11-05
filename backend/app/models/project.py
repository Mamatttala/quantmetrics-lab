"""Named workspaces for datasets and their derived artifacts."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class Project(BaseModel):
    project_id: str
    name: str
    created_at: datetime


class CreateProject(BaseModel):
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Enter a project name.")
        if any(ord(character) < 32 for character in value):
            raise ValueError("Project names cannot contain control characters.")
        return value
