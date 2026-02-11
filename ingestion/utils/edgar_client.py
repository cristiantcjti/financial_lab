from typing import Any

from edgar import Company, set_identity


class EdgarClient:
    FORM_ITEMS = {
        "10-K": ["1", "1A", "7", "8", "9A"],
        "10-Q": ["1", "2", "3", "4"],
    }

    def __init__(self, email: str) -> None:
        set_identity(email)

    def fetch_filing_data(self, ticker: str, form_type: str) -> dict[str, Any]:
        company = Company(ticker)
        filing = company.get_filings(form=form_type).latest()

        if filing is None:
            raise ValueError(f"No {form_type} filing found for {ticker}")

        metadata = {
            "ticker": ticker,
            "company_name": filing.company,  # type: ignore[attr-defined]
            "report_date": str(filing.report_date),  # type: ignore[attr-defined]
            "form_type": filing.form,  # type: ignore[attr-defined]
        }

        filing_obj = filing.obj()  # type: ignore[attr-defined]
        if filing_obj is None:
            raise ValueError(f"Could not retrieve filing object for {ticker}")

        items = {}

        for item_num in self.FORM_ITEMS[form_type]:
            item_key = f"Item {item_num}"
            try:
                items[item_key] = filing_obj[item_key]  # type: ignore[index]
            except (KeyError, IndexError):
                continue

        return {"metadata": metadata, "items": items}

    def get_combined_text(self, data: dict) -> str:
        texts = []
        for item_name, item_content in data["items"].items():
            texts.append(f"## {item_name}\n\n{item_content}")

        return "\n\n".join(texts)
