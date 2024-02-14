
default_helptext = ""

models_helptext = {
       "juggernautXL_v8Rundiffusion.safetensors": "default Fooocus model.",
       #...add any new models here to include a help text to show in the UI
}

def get_html(text):
     return f'<p style="padding: 4px 8px;">{text}</p>'

def update_help_text(selected_model):
    return get_html(models_helptext.get(selected_model, default_helptext))