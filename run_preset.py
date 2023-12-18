import os
import subprocess


def list_presets():
    preset_folder = 'presets'
    return [f for f in os.listdir(preset_folder) if f.endswith('.json')]


def display_presets(presets):
    print("Available presets:")
    for i, preset in enumerate(presets, start=1):
        print(f"{i}. {os.path.splitext(preset)[0]}")


def select_preset():
    presets = list_presets()

    if not presets:
        print("No presets found. Launching with default config")
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

    command = [python_executable, '-s', script_path]

    if selected_preset:
        command += ['--preset', selected_preset]

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing the script: {e}")


if __name__ == "__main__":
    selected_preset = select_preset()
    run_with_selected_preset(selected_preset)
