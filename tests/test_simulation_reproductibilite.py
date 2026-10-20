"""Régressions exécutables avec Python 3.12/3.13, sans dépendance scientifique."""
import ast
import json
from math import factorial
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]
SIMULATION = ROOT / 'analyse/SIMULATION_2026.ipynb'
MODELE = ROOT / 'analyse/MODEL_POISSON.ipynb'


def lire_notebook(path):
    return json.loads(path.read_text(encoding='utf-8'))


def definitions(path):
    noeuds = []
    for cell in lire_notebook(path)['cells']:
        if cell['cell_type'] == 'code':
            noeuds.extend(ast.parse(''.join(cell['source'])).body)
    return noeuds


class ReproductibiliteSimulation(unittest.TestCase):
    def setUp(self):
        self.espace = {'ast': ast, 'json': json, 'Path': Path}
        fonction = next(n for n in definitions(SIMULATION)
                        if isinstance(n, ast.FunctionDef) and n.name == 'extraire_fonctions_notebook')
        exec(compile(ast.Module(body=[fonction], type_ignores=[]), '<test>', 'exec'), self.espace)

    def test_lecture_utf8_avec_defaut_windows_cp1252_et_crlf(self):
        extraire = self.espace['extraire_fonctions_notebook']
        attendu = {nom: ast.dump(n) for nom, n in extraire(MODELE).items()}
        original = Path.read_text

        def lecture_windows(path, *args, encoding=None, **kwargs):
            return original(path, *args, encoding=encoding or 'cp1252', **kwargs)

        with tempfile.TemporaryDirectory() as dossier:
            copie = Path(dossier) / 'modèle_équipe.ipynb'
            copie.write_bytes(MODELE.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
            with patch.object(Path, 'read_text', lecture_windows):
                obtenu = {nom: ast.dump(n) for nom, n in extraire(copie).items()}
            self.assertEqual(attendu, obtenu)

    def test_comparaison_des_definitions_dans_le_meme_interpreteur(self):
        extraire = self.espace['extraire_fonctions_notebook']
        modele = extraire(MODELE)
        simulation = extraire(SIMULATION)
        self.assertEqual(set(modele), set(simulation))
        self.assertEqual({nom: ast.dump(n) for nom, n in modele.items()},
                         {nom: ast.dump(n) for nom, n in simulation.items()})
        source = SIMULATION.read_text(encoding='utf-8')
        self.assertNotIn('324fc1caba11d0c66403bb24dd8b89a5493bf62ae930dbe4748e85f8e65863df', source)

    def test_toutes_les_lectures_et_ecritures_textuelles_precisent_encodage(self):
        appels = [n for racine in definitions(SIMULATION) for n in ast.walk(racine)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr in {'read_text', 'write_text'}]
        self.assertGreaterEqual(len(appels), 2)
        for appel in appels:
            encodage = next((k.value for k in appel.keywords if k.arg == 'encoding'), None)
            self.assertIsInstance(encodage, ast.Constant)
            self.assertEqual(encodage.value, 'utf-8')

    def test_definition_dupliquee_est_refusee(self):
        contenu = lire_notebook(MODELE)
        contenu['cells'].append({'cell_type': 'code', 'source': ['def tirer_score():\n', '    pass\n']})
        with tempfile.TemporaryDirectory() as dossier:
            path = Path(dossier) / 'modele.ipynb'
            path.write_text(json.dumps(contenu, ensure_ascii=False), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'plusieurs fois'):
                self.espace['extraire_fonctions_notebook'](path)

    def test_complement_finale_separe_nul_et_vainqueur(self):
        affectation = next(n for n in definitions(SIMULATION)
                           if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'FINALES_COMPLEMENTAIRES'
                                   for t in n.targets))
        espace = {}
        exec(compile(ast.Module(body=[affectation], type_ignores=[]), '<test>', 'exec'), espace)
        finale = espace['FINALES_COMPLEMENTAIRES'][2025]
        self.assertEqual(finale['date'], '2026-05-30')
        self.assertEqual((finale['buts_domicile_90'], finale['buts_exterieur_90']), (1, 1))
        self.assertEqual((finale['tirs_au_but_domicile'], finale['tirs_au_but_exterieur']), (4, 3))
        self.assertEqual(finale['vainqueur_club_id'], 583)
        self.assertEqual(finale['usage'], 'evaluation_uniquement')
        self.assertNotIn('game_id', finale)

    @unittest.skipIf(np is None, 'Test numérique : NumPy non installé dans cet interpréteur.')
    def test_contrat_numerique_et_detection_d_un_vrai_ecart(self):
        fonction = next(n for n in definitions(SIMULATION)
                        if isinstance(n, ast.FunctionDef) and n.name == 'verifier_compatibilite_modele')
        self.espace.update(np=np, factorial=factorial, GRAINE=2026)
        exec(compile(ast.Module(body=[fonction], type_ignores=[]), '<test>', 'exec'), self.espace)
        local = {'np': np, 'factorial': factorial}
        noeuds = self.espace['extraire_fonctions_notebook'](SIMULATION)
        exec(compile(ast.Module(body=list(noeuds.values()), type_ignores=[]), '<test>', 'exec'), local)
        parametres = {'intercept': .3, 'coef_ecart_100': .2, 'coef_terrain': .1,
                      'coef_nouveau_format': .1, 'max_buts': 10}
        verifier = self.espace['verifier_compatibilite_modele']
        bilan = verifier(MODELE, SIMULATION, parametres, local)
        self.assertEqual(bilan['cas_valides'], 282)
        fonctions_erronees = dict(local)
        fonctions_erronees['buts_attendus'] = lambda *args: (999999., 999999.)
        with self.assertRaisesRegex(AssertionError, 'buts_attendus diffère'):
            verifier(MODELE, SIMULATION, parametres, fonctions_erronees)


if __name__ == '__main__':
    unittest.main()
