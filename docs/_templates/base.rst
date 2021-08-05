{{ fullname | replace(module.split(".")[-1] + ".", "") }}
{{ underline }}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
