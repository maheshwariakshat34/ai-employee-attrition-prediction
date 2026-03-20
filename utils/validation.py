REQUIRED_FIELDS = [
    "Age", "TotalWorkingYears", "YearsInCurrentRole", "YearsWithCurrManager",
    "JobLevel", "MonthlyIncome", "StockOptionLevel",
    "OverTime_Yes", "MaritalStatus_Single", "JobRole_Sales Representative"
]


def validate_input(data):
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in data or data.get(field) in (None, ""):
            errors.append(f"{field!r} is required")

    if errors:
        return errors

    age = None
    total_years = None
    years_role = None
    manager_years = None

    # Age
    try:
        age = int(data.get("Age"))
        if age < 18 or age > 65:
            errors.append("Age must be between 18 and 65")
    except (TypeError, ValueError):
        errors.append("Age must be a whole number")

    # TotalWorkingYears
    try:
        total_years = float(data.get("TotalWorkingYears"))
        if total_years < 0 or total_years > 40:
            errors.append("TotalWorkingYears must be between 0 and 40")
    except (TypeError, ValueError):
        errors.append("TotalWorkingYears must be numeric")

    # YearsInCurrentRole
    try:
        years_role = float(data.get("YearsInCurrentRole"))
        if years_role < 0:
            errors.append("YearsInCurrentRole cannot be negative")
    except (TypeError, ValueError):
        errors.append("YearsInCurrentRole must be numeric")

    # YearsWithCurrManager
    try:
        manager_years = float(data.get("YearsWithCurrManager"))
        if manager_years < 0:
            errors.append("YearsWithCurrManager cannot be negative")
    except (TypeError, ValueError):
        errors.append("YearsWithCurrManager must be numeric")

    # JobLevel
    try:
        job_level = int(data.get("JobLevel"))
        if job_level < 1 or job_level > 5:
            errors.append("JobLevel must be between 1 and 5")
    except (TypeError, ValueError):
        errors.append("JobLevel must be a whole number between 1 and 5")

    # MonthlyIncome
    try:
        monthly_income = float(data.get("MonthlyIncome"))
        if monthly_income < 0:
            errors.append("MonthlyIncome cannot be negative")
    except (TypeError, ValueError):
        errors.append("MonthlyIncome must be numeric")

    # StockOptionLevel
    try:
        stock_option = int(data.get("StockOptionLevel"))
        if stock_option < 0 or stock_option > 3:
            errors.append("StockOptionLevel must be between 0 and 3")
    except (TypeError, ValueError):
        errors.append("StockOptionLevel must be a whole number between 0 and 3")

    # Cross-field logic
    if age is not None and total_years is not None:
        if total_years >= age:
            errors.append("TotalWorkingYears must be less than Age")
        if (age - total_years) < 18:
            errors.append("TotalWorkingYears is unrealistic for the given Age")

    if total_years is not None and years_role is not None:
        if years_role > total_years:
            errors.append("YearsInCurrentRole cannot exceed TotalWorkingYears")

    if years_role is not None and manager_years is not None:
        if manager_years > years_role:
            errors.append("YearsWithCurrManager cannot exceed YearsInCurrentRole")

    return errors