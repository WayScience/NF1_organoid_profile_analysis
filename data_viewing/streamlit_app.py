"""Entry point some Streamlit hosts look for by this exact filename.

The real app lives in ``app.py`` (page config, sidebar filters, tabs); this
file only exists so a host that defaults to/expects ``streamlit_app.py``
(e.g. Streamlit Community Cloud) finds something to run.

This re-execs app.py's source directly rather than ``from app import *``:
Streamlit reruns this file on every widget interaction, but a plain import
only runs a module's top-level code once (Python caches it in
``sys.modules``), so a later rerun would silently do nothing. ``exec`` has
no such cache -- it matches how Streamlit runs the main script itself.
"""

import pathlib

_app_path = pathlib.Path(__file__).with_name("app.py")
exec(compile(_app_path.read_text(), str(_app_path), "exec"))
