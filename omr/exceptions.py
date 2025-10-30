class FileFormatNotSupportedError(ValueError):
    """Raised when an unsupported file format is encountered."""

    def __init__(self, filename: str, supported_formats: list[str] | None = None):
        self.filename = filename
        self.supported_formats = supported_formats or []
        ext = filename.split(".")[-1] if "." in filename else None
        message = f"File format '{ext}' from '{filename}' is not supported." + (
            f" Supported extensions: {', '.join(self.supported_formats)}."
            if self.supported_formats
            else ""
        )
        super().__init__(message)
