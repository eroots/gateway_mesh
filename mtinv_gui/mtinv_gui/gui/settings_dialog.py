"""Machine-level settings: paths to deal.II / pyGoFEM and the OpenTopography key."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog,
)

from .. import config


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(620, 220)
        data = config.load_settings()

        form = QFormLayout(self)
        hdr = QLabel(f"Stored in <code>{config.SETTINGS_PATH}</code> "
                     "(this file overrides the defaults in config.py).")
        hdr.setWordWrap(True)
        hdr.setStyleSheet("color: gray;")
        form.addRow(hdr)
        self.fields: dict[str, QLineEdit] = {}
        for key, label, is_dir in (
            ("dealii_path", "deal.II path (dir with PyDealII/)", True),
            ("gofem_path", "pyGoFEM repo path", True),
            ("mtpy_path", "mtpy source path (optional)", True),
            ("opentopo_api_key", "OpenTopography API key", False),
        ):
            edit = QLineEdit(data.get(key, ""))
            self.fields[key] = edit
            if is_dir:
                row = QHBoxLayout()
                row.addWidget(edit, 1)
                btn = QPushButton("...")
                btn.setFixedWidth(30)
                btn.clicked.connect(lambda _, e=edit: self._pick_dir(e))
                row.addWidget(btn)
                form.addRow(label, row)
            else:
                form.addRow(label, edit)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _pick_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
        if d:
            edit.setText(d)

    def _save(self):
        config.save_settings({k: e.text().strip() for k, e in self.fields.items()})
        self.accept()
