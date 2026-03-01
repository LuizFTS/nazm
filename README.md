# Nazm

**Nazm** is a Python library for reliable desktop automation using visual detection instead of fragile screen coordinates.

It provides a clean, high-level API for interacting with legacy desktop applications by locating UI elements through image recognition and performing actions like clicking, typing, and waiting.

---

## Why Nazm?

Traditional GUI automation scripts break easily because they rely on fixed positions.
Nazm focuses on **visual anchoring**, making automations more resilient and reusable.

Key goals:

- Image-based UI detection
- Clean automation API
- Reliable retries and waits
- Minimal configuration
- Designed for real production workflows

---

## Installation

```bash
pip install nazm
```

---

## Quick example

```python
import nazm

nazm.wait_for("login_button.png")
nazm.click("login_button.png")
nazm.type_into("username_field.png", "admin")
nazm.type_into("password_field.png", "secret")
nazm.click("submit.png")
```

---

## Status

Nazm is currently in early development.
The initial release focuses on a stable core engine for visual automation.

---

## License

MIT
