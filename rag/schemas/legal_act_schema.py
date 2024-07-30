from typing import List, Optional
from pydantic import BaseModel, field_validator
import re

class LegalActSchema(BaseModel):
    act_id: int # Numeric ID for the legal act
    act_number: str # Unique identifier for the legal act
    act_year: str # Year of the legal act
    ministry: str # Ministry responsible for the legal act
    department: str # Department responsible for the legal act
    enactment_date: str # Date when the legal act was enacted
    enforcement_date: str # Date when the legal act was enforced
    short_title: str # Short title of the legal act
    long_title: str # Long title of the legal act
    pdf_file_path: str # Path to the PDF file of the legal act
    text_content: str # Text content of the legal act (extracted from the PDF)
    last_updated: str # Date when the legal act entry was last udpated
    tags: Optional[List[str]] = None
    embedding: Optional[List[float]] = None

    @field_validator('act_id', mode='before')
    def validate_act_id(cls, value):
        if not (0 < value <= 999999999):
            raise ValueError("act_id must be between 1 and 999,999,999")
        return value

    @field_validator('act_year', mode='before')
    def validate_act_year(cls, value):
        if not re.match(r'^\d{4}$', value):
            raise ValueError("act_year must be a 4-digit year")
        return value

    @field_validator('enactment_date', 'enforcement_date', 'last_updated', mode='before')
    def validate_date_format(cls, value):
        # Assuming date format is YYYY-MM-DD
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            raise ValueError("Date must be in the format YYYY-MM-DD")
        return value

    @field_validator('pdf_file_path', mode='before')
    def validate_pdf_file_path(cls, value):
        # Basic check: Ensure the path ends with '.pdf'
        if not value.lower().endswith('.pdf'):
            raise ValueError("pdf_file_path must end with '.pdf'")
        return value

    @field_validator('text_content', mode='before')
    def validate_text_content(cls, value):
        if len(value) < 10:
            raise ValueError("text_content must be at least 10 characters long")
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