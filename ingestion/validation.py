"""
We use @field_validator to define custom validation functions for the schema fields, but there are still reasons why we might need validation.py:

When You Might Still Need validation.py:
1. Complex Validation Logic:
If you have complex validation logic that goes beyond simple field validation (e.g., cross-field validation or complex business rules), it might make sense to handle this in validation.py.

2. Custom Validation Functions:
If you need reusable custom validation functions that apply to multiple schemas or parts of your application, placing them in validation.py can help keep your code organized.

3. Validation Beyond Schema:
For operations like checking the validity of external data sources or custom logic that isn't directly related to field-level validation, validation.py might still be relevant.
Historical Reasons:

If your project has a history of using validation.py for validation and the current schema validations are a recent addition, you might temporarily retain validation.py while migrating the existing validation logic.
"""