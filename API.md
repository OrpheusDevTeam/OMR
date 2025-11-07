# Orpheus OMR CLI – Output Specification

## Overview

The **Orpheus** CLI tool processes one or more image or PDF files and outputs a **JSON-formatted** result indicating success or failure.  
All results are printed to **stdout**.

---

## ✅ Success Response

**Condition:**  
All input paths are valid, and the tool successfully processes them into a MusicXML file.

**JSON Schema:**

```json
{
  "status": "success",
  "filepath": "<absolute_path_to_output_musicxml>"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always  `"success"`. |
| `filepath` | string | Absolute path to the generated MusicXML file. |

**Example:**

```json
{
  "status": "success",
  "filepath": "/home/user/project/output.musicxml"
}
```

---

## ❌ Error Response

**Condition:**  
Returned when any exception occurs during execution (invalid path, unsupported format, etc.).

**JSON Schema:**

```json
{
  "status": "error",
  "error_type": "<ExceptionClassName>",
  "message": "<human_readable_message>"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"error"`. |
| `error_type` | string | The class name of the raised exception. |
| `message` | string | Human-readable error message describing the cause. |

**Example:**

```json
{
  "status": "error",
  "error_type": "FileFormatNotSupportedError",
  "message": "The provided file format '.tiff' is not supported."
}
```

---

## ⚠️ Known Error Types

| `error_type` | Description | Exit Code |
|---------------|-------------|------------|
| `FileFormatNotSupportedError` | The provided file(s) are of an unsupported type. | `EXIT_UNSUPPORTED_FORMAT` |
| `FileNotFoundError` | None of the provided paths exist or are accessible. | `EXIT_GENERIC_ERROR` |
| `KeyboardInterrupt` | Execution was manually stopped by the user. | `EXIT_KEYBOARD_INTERRUPT` |
| `<other>` | Any other unhandled Python exception. | `EXIT_GENERIC_ERROR` |

---

## 🧾 Exit Codes

| Constant | Value | Meaning |
|-----------|--------|---------|
| `EXIT_SUCCESS` | 0 | Execution completed successfully. |
| `EXIT_GENERIC_ERROR` | 1 | A general error occurred. |
| `EXIT_UNSUPPORTED_FORMAT` | 2 | Input format not supported. |
| `EXIT_KEYBOARD_INTERRUPT` | 130 | Process interrupted by user. |

---

## 📘 Notes

- All logs (info/warning/error) are written through the internal logger and **do not** affect JSON output (they go to `stderr`).  
- The CLI guarantees a valid JSON object on every run.  
- The generated MusicXML file is always named `output.musicxml` and saved in the current working directory.
