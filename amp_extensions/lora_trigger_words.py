
loras_trigger_words = {
     #   "sd_xl_offset_example-lora_1.0.safetensors": "example lora trigger word",
       #...add any new loras here to include the trigger words to show in the UI
}

def get_trigger_word(selected_lora, current_prompt):
    return loras_trigger_words.get(selected_lora, current_prompt)