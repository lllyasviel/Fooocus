import os
import subprocess


def list_presets():
    preset_folder = 'presets'
    presets = [f for f in os.listdir(preset_folder) if f.endswith('.json')]
    return presets


def display_presets(presets):
    print("Available presets:")
    for i, preset in enumerate(presets, start=1):
        print(f"{i}. {os.path.splitext(preset)[0]}")


def select_preset():
    presets = list_presets()
    
    if not presets:
        print("No presets found.")
        return None
    
    display_presets(presets)
    
    while True:
        try:
            user_input = input("Select the preset to launch: ")
            selected_index = int(user_input)
            
            if 1 <= selected_index <= len(presets):
                return os.path.splitext(presets[selected_index - 1])[0]
            
            print("Invalid input. Please enter a valid number.")

        except ValueError:
            print("Invalid input. Please enter a number.")


def run_with_selected_preset(selected_preset):
    script_path = os.path.abspath('entry_with_update.py')
    python_executable = os.path.abspath('../python_embeded/python.exe')

    if os.path.exists(script_path):
        command = [python_executable, '-s', script_path, '--preset', selected_preset]
        
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing the script: {e}")
    else:
        print("Script not found.")


if __name__ == "__main__":
    selected_preset = select_preset()

    if selected_preset:
        run_with_selected_preset(selected_preset)
