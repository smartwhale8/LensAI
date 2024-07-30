from pydantic import BaseModel, field_validator
from typing import List, Optional
import re

class CaseFileSchema(BaseModel):
    case_id: int
    case_year: str
    case_title: str
    plaintiff: str
    defendant: str
    filing_date: str
    resolution_date: str
    case_summary: str
    pdf_file_path: str
    text_content: str
    last_updated: str
    tags: Optional[List[str]] = None
    embedding: Optional[List[float]] = None

    @field_validator('case_id', mode='before')
    def validate_case_id(cls, value):
        if not (0 < value <= 999999999):
            raise ValueError("case_id must be between 1 and 999,999,999")
        return value

    @field_validator('case_year', mode='before')
    def validate_case_year(cls, value):
        if not re.match(r'^\d{4}$', value):
            raise ValueError("case_year must be a 4-digit year")
        return value

    @field_validator('filing_date', 'resolution_date', 'last_updated', mode='before')
    def validate_date_format(cls, value):
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            raise ValueError("Date must be in the format YYYY-MM-DD")
        return value

    @field_validator('pdf_file_path', mode='before')
    def validate_pdf_file_path(cls, value):
        if not value.lower().endswith('.pdf'):
            raise ValueError("pdf_file_path must end with '.pdf'")
        return value

    @field_validator('text_content', mode='before')
    def validate_text_content(cls, value):
        if len(value) < 10:
            raise ValueError("text_content must be at least 10 characters long")
        return value

    @field_validator('case_summary', mode='before')
    def validate_case_summary(cls, value):
        if len(value) < 20:
            raise ValueError("case_summary must be at least 20 characters long")
        return value

    @field_validator('tags', mode='before')
    def validate_tags(cls, value):
        if value is not None and any(not isinstance(tag, str) for tag in value):
            raise ValueError("All tags must be strings")
        return value

    @field_validator('embedding', mode='before')
    def validate_embedding(cls, value):
        if value is not None:
            if not all(isinstance(val, float) for val in value):
                raise ValueError("All elements in embedding must be floats")
            if len(value) != 768:
                raise ValueError("embedding must be a list of 768 floats")
        return value
