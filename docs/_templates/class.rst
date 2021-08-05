{{ fullname | replace(module.split(".")[-1] + ".", "") }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :special-members:
   :inherited-members:

   {% block methods %}

      {% if methods %}
         .. rubric:: {{ _('Methods') }}

         .. autosummary::
            :nosignatures:
            {% for item in methods %}
               ~{{ name }}.{{ item }}
            {%- endfor %}
      {% endif %}
   {% endblock %}
