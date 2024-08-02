import subprocess
import sys
import os
import json
import glob

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
        # Affiche la sortie du script
        print("Sortie du script :\n", result.stdout)
        if result.stderr:
            print("Erreurs du script :\n", result.stderr)

        # Lecture du fichier JSON généré par le script shell
        json_file_prefix = 'simplecnn'  # Le préfixe que tu utilises dans save_results_to_json
        result_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AbstractRay', 'result', json_file_prefix)
        json_files = glob.glob(os.path.join(result_folder, '*.json'))
        if json_files:
            # Suppose que le fichier le plus récent est celui que nous venons de créer
            latest_json_file = max(json_files, key=os.path.getctime)
            with open(latest_json_file, 'r') as file:
                json_output = json.load(file)
            return json_output
        else:
            print("Aucun fichier JSON trouvé.")
            return None

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
            if isinstance(output, dict):
                print("La sortie JSON du script est :")
                print(json.dumps(output, indent=4))
            else:
                print("La sortie brute du script est :\n", output)
