import ast
import sys
import argparse
import os

# Versuche, Standardbibliotheksmodule für die aktuelle Python-Version zu ermitteln
try:
    # Verfügbar ab Python 3.10
    from sys import stdlib_module_names as STDLIBS
except ImportError:
    # Fallback für ältere Versionen (Liste ist nicht vollständig!)
    # Du könntest hier eine umfangreichere Liste pflegen oder eine externe Bibliothek wie 'is_stdlib' verwenden
    print("Warnung: sys.stdlib_module_names nicht verfügbar (Python < 3.10). Standardbibliothek-Erkennung ist begrenzt.", file=sys.stderr)
    STDLIBS = {
        'os', 'sys', 'math', 'json', 'datetime', 'time', 're', 'collections',
        'io', 'argparse', 'logging', 'subprocess', 'multiprocessing', 'threading',
        'socket', 'select', 'struct', 'pickle', 'shutil', 'tempfile', 'glob',
        'random', 'itertools', 'functools', 'operator', 'typing', 'pathlib',
        'urllib', 'http', 'email', 'csv', 'configparser', 'sqlite3', 'xml',
        'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'hashlib', 'hmac', 'secrets',
        'uuid', 'warnings', 'abc', 'contextlib', 'enum', 'types', 'weakref',
        'copy', 'pprint', 'decimal', 'fractions', 'statistics'
        # Füge hier bei Bedarf weitere häufig verwendete Standardmodule hinzu
    }
    # Füge Built-in-Module hinzu (immer verfügbar)
    STDLIBS.update(sys.builtin_module_names)


class ImportFinder(ast.NodeVisitor):
    """
    Durchläuft einen Abstract Syntax Tree (AST) und sammelt importierte Modulnamen.
    """
    def __init__(self):
        self.imports = set()
        self.stdlib_modules = STDLIBS

    def visit_Import(self, node):
        for alias in node.names:
            # Nimm den Modulnamen (ignoriere Aliase wie 'import requests as r')
            module_name = alias.name.split('.')[0] # Nur den Top-Level-Namen (z.B. 'os' aus 'os.path')
            if module_name not in self.stdlib_modules:
                self.imports.add(module_name)
        self.generic_visit(node) # Besuche auch Kindknoten

    def visit_ImportFrom(self, node):
        # Ignoriere relative Imports (z.B. 'from . import utils')
        if node.level > 0:
            self.generic_visit(node)
            return

        if node.module:
             # Nimm den Modulnamen (z.B. 'pandas' aus 'from pandas import DataFrame')
            module_name = node.module.split('.')[0] # Nur den Top-Level-Namen
            if module_name not in self.stdlib_modules:
                self.imports.add(module_name)
        self.generic_visit(node) # Besuche auch Kindknoten

def find_modules_in_file(filepath):
    """
    Liest eine Python-Datei, parst sie und extrahiert externe Modulimporte.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Fehler: Datei nicht gefunden: {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Fehler beim Lesen der Datei {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        tree = ast.parse(content, filename=filepath)
        finder = ImportFinder()
        finder.visit(tree)
        return sorted(list(finder.imports)) # Sortiert für konsistente Ausgabe
    except SyntaxError as e:
        print(f"Fehler: Syntaxfehler in der Datei {filepath}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Fehler beim Parsen der Datei {filepath}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Ermittelt benötigte externe Python-Module aus einer .py-Datei und gibt den pip install-Befehl aus.")
    parser.add_argument("python_file", help="Der Pfad zur .py-Datei, die analysiert werden soll.")
    args = parser.parse_args()

    if not os.path.exists(args.python_file):
         print(f"Fehler: Datei nicht gefunden: {args.python_file}", file=sys.stderr)
         sys.exit(1)

    if not args.python_file.lower().endswith('.py'):
         print(f"Warnung: Die angegebene Datei '{args.python_file}' scheint keine .py-Datei zu sein.", file=sys.stderr)


    required_modules = find_modules_in_file(args.python_file)

    if not required_modules:
        print("Keine externen Modulimporte gefunden (oder nur Standardbibliotheksmodule).")
    else:
        install_command = f"pip install {' '.join(required_modules)}"
        print("\n--- Copy & Paste Installationsbefehl ---")
        print(install_command)
        print("----------------------------------------")
        print("\nGefundene externe Module:")
        for mod in required_modules:
            print(f"- {mod}")

    print("\nHinweis: Dieses Skript erkennt nur die Namen der importierten Module.")
    print("Es ermittelt keine spezifischen Versionen und löst keine Abhängigkeiten von Abhängigkeiten auf.")
    print("Für komplexes Abhängigkeitsmanagement sind Werkzeuge wie pip freeze, pipreqs oder Poetry besser geeignet.")


if __name__ == "__main__":
    main()
