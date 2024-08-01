import subprocess
import sys
import os

def execute_script(script_path):
    # Vérifie si le fichier existe
    if not os.path.isfile(script_path):
        print(f"Le fichier {script_path} n'existe pas.")
        return None

    # Vérifie si c'est bien un script shell
    if not script_path.endswith('.sh'):
        print(f"Le fichier {script_path} n'est pas un script shell.")
        return None

    # Change les permissions pour rendre le script exécutable
    os.chmod(script_path, 0o755)

    try:
        # Exécute le script en capturant la sortie
        result = subprocess.run(['bash', script_path], check=True, capture_output=True, text=True)
        print("Sortie du script :\n", result.stdout)
        if result.stderr:
            print("Erreurs du script :\n", result.stderr)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution du script {script_path}: {e}")
        print(f"Sortie standard :\n{e.stdout}")
        print(f"Erreur standard :\n{e.stderr}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python execute_script.py <path_to_script.sh>")
    else:
        script_path = sys.argv[1]
        output = execute_script(script_path)
        if output is not None:
            print(output)
