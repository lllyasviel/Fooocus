import os
import subprocess


def list_presets():
    preset_folder = 'presets'
    if not os.path.exists(preset_folder):
        print("No presets found. Launching with default config")
        return None

    presets = [f for f in os.listdir(preset_folder) if f.endswith('.json')]
    return presets if presets else None


def display_presets(presets):
    print("Available presets:")
    for i, preset in enumerate(presets, start=1):
        print(f"{i}. {os.path.splitext(preset)[0]}")


def select_preset():
    presets = list_presets()

    if presets is None:
        return None

    display_presets(presets)

    while True:
        user_input = input("Select a preset to launch: ")
        try:
            selected_index = int(user_input)
            if 1 <= selected_index <= len(presets):
                return os.path.splitext(presets[selected_index - 1])[0]
        except ValueError:
            pass
        print(f"Invalid input. Please enter a number between 1 and {len(presets)}")


def run_with_selected_preset(selected_preset):
    script_path = 'entry_with_update.py'
    python_executable = os.path.join('..', 'python_embeded', 'python.exe')

    command = [python_executable, '-s', script_path]

    if selected_preset:
        command += ['--preset', selected_preset]

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing the script: {e}")


if __name__ == "__main__":
    try:
        selected_preset = select_preset()
        if selected_preset is not None:
            run_with_selected_preset(selected_preset)
        else:
            run_with_selected_preset(None)
    except FileNotFoundError as e:
        print(f"Error: {e}")
