# Checklist & Verification Files

This folder contains helper scripts and documentation for setup verification and testing.

## Files

### Verification Scripts

- **`verify_data.py`** - Verifies data loading and configuration
  ```bash
  python checklist/verify_data.py
  ```

- **`test_imports.py`** - Tests all package imports
  ```bash
  python checklist/test_imports.py
  ```

### Documentation

- **`SETUP_VERIFICATION.md`** - Detailed setup verification report
- **`CONFIG_UPDATE_SUMMARY.md`** - Configuration changes summary
- **`QUICK_START.md`** - Quick start guide
- **`QUICK_REFERENCE.md`** - Quick reference card

## Usage

These files are for verification and reference only. For actual training, use the main script:

```bash
python main.py
```

## When to Use These Files

- **After initial setup** - Run `verify_data.py` and `test_imports.py`
- **When troubleshooting** - Check the documentation files
- **When updating configuration** - Refer to `CONFIG_UPDATE_SUMMARY.md`
- **For quick reference** - Use `QUICK_REFERENCE.md`
