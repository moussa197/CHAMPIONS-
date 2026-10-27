from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


DOSSIER = Path(__file__).resolve().parent.parent / "output"
FICHIERS = {
    "elo": "elo_participants_2026.csv",
    "calendrier": "calendrier_2026.csv",
    "predictions": "predictions_phase_ligue_2026.csv",
    "simulation": "resultats_simulation_2026.csv",
    "modele": "modele_scores.joblib",
    "forces": "forces_participants_2026.csv",
}
MANQUANTS = {
    "elo": "Exécutez `analyse/v2_elo.ipynb` pour produire l’Elo des participants.",
    "calendrier": "Exécutez `analyse/v2_participants.ipynb` pour produire le calendrier UEFA.",
    "predictions": (
        "Exécutez `analyse/MODEL_POISSON.ipynb` après les notebooks de préparation des "
        "données pour générer ce fichier."
    ),
    "modele": (
        "Exécutez `analyse/MODEL_POISSON.ipynb` après les notebooks de préparation des "
        "données pour générer ce modèle."
    ),
    "forces": (
        "Exécutez `analyse/v2_forces.ipynb` pour produire les forces d’attaque et de défense "
        "des participants."
    ),
}


@st.cache_data
def lire_csv(nom_fichier: str) -> pd.DataFrame:
    return pd.read_csv(DOSSIER / nom_fichier)


@st.cache_resource
def charger_modele(chemin: str) -> Any:
    return joblib.load(chemin)


def _normaliser(nom: str) -> str:
    sans_accents = unicodedata.normalize("NFKD", str(nom)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", sans_accents.lower())


def entrees_match(forces: pd.DataFrame, club_dom: int, club_ext: int) -> dict[str, float]:
    """Entrées du modèle v3 pour « club_dom reçoit club_ext » (forces indexées par club_id)."""
    a, b = forces.loc[club_dom], forces.loc[club_ext]
    return {
        "force_ad_dom": float(a["attaque"] + b["defense"]),
        "force_ad_ext": float(b["attaque"] + a["defense"]),
        "ecart_valeur_recente": float(
            np.log(a["valeur_recente_eur"]) - np.log(b["valeur_recente_eur"])
        ),
    }


def buts_attendus(
    parametres: dict[str, Any],
    ecart_elo: float,
    domicile: int = 1,
    minutes: int = 90,
    nouveau_format: int = 1,
    force_ad_dom: float | None = None,
    force_ad_ext: float | None = None,
    ecart_valeur_recente: float = 0.0,
) -> tuple[float, float]:
    # Sans forces connues, chaque équipe prend la force médiane de l'apprentissage
    f_dom = parametres["mediane_force_ad"] if force_ad_dom is None else force_ad_dom
    f_ext = parametres["mediane_force_ad"] if force_ad_ext is None else force_ad_ext
    niveau = parametres["intercept"] + parametres["coef_nouveau_format"] * nouveau_format
    avantage = (
        parametres["coef_ecart_100"] * ecart_elo / 100
        + parametres["coef_terrain"] * domicile
        + parametres["coef_valeur_recente"] * ecart_valeur_recente
    )
    duree = minutes / 90
    return (
        float(np.exp(niveau + avantage + parametres["coef_force_ad"] * f_dom) * duree),
        float(np.exp(niveau - avantage + parametres["coef_force_ad"] * f_ext) * duree),
    )


def probas_match(
    parametres: dict[str, Any], ecart_elo: float, domicile: int = 1, **entrees: float
) -> dict[str, float]:
    scores = matrice_scores(parametres, ecart_elo, domicile, **entrees)
    return {
        "Victoire domicile": float(np.tril(scores, -1).sum()),
        "Nul": float(np.trace(scores)),
        "Victoire extérieur": float(np.triu(scores, 1).sum()),
    }


def matrice_scores(
    parametres: dict[str, Any],
    ecart_elo: float,
    domicile: int = 1,
    minutes: int = 90,
    **entrees: float,
) -> np.ndarray:
    buts_dom, buts_ext = buts_attendus(parametres, ecart_elo, domicile, minutes, **entrees)
    buts = np.arange(int(parametres["max_buts"]) + 1)
    factorielles = np.array([math.factorial(int(n)) for n in buts], dtype=float)
    matrice = np.outer(
        np.exp(-buts_dom) * buts_dom**buts / factorielles,
        np.exp(-buts_ext) * buts_ext**buts / factorielles,
    )
    return matrice / matrice.sum()


def tirer_score(
    parametres: dict[str, Any],
    ecart_elo: float,
    domicile: int = 1,
    minutes: int = 90,
    rng: np.random.Generator | None = None,
    **entrees: float,
) -> tuple[int, int]:
    rng = rng if rng is not None else np.random.default_rng()
    buts_dom, buts_ext = buts_attendus(parametres, ecart_elo, domicile, minutes, **entrees)
    return int(rng.poisson(buts_dom)), int(rng.poisson(buts_ext))


def scores_les_plus_probables(
    scores: np.ndarray, nombre: int = 5
) -> list[tuple[int, int, float]]:
    """Les `nombre` scores les plus probables : (buts domicile, buts extérieur, probabilité)."""
    ordre = np.argsort(scores, axis=None)[::-1][:nombre]
    return [
        (int(i), int(j), float(scores[i, j]))
        for i, j in zip(*np.unravel_index(ordre, scores.shape))
    ]


def _forces_par_club(forces: pd.DataFrame) -> pd.DataFrame:
    requis = {"club_id", "attaque", "defense", "valeur_recente_eur"}
    if not requis.issubset(forces.columns):
        raise ValueError(
            "`forces_participants_2026.csv` doit contenir les colonnes "
            + ", ".join(sorted(requis))
            + "."
        )
    return forces.set_index("club_id")


def _colonne(df: pd.DataFrame, noms: tuple[str, ...]) -> str | None:
    disponibles = {_normaliser(colonne): colonne for colonne in df.columns}
    for nom in noms:
        if _normaliser(nom) in disponibles:
            return disponibles[_normaliser(nom)]
    return None


def _erreur_fichier(cle: str) -> None:
    st.error(f"Fichier manquant : `output/{FICHIERS[cle]}`. {MANQUANTS[cle]}")


def _charger_csv_requis(cle: str) -> pd.DataFrame | None:
    chemin = DOSSIER / FICHIERS[cle]
    if not chemin.is_file():
        _erreur_fichier(cle)
        return None
    return lire_csv(FICHIERS[cle])


def _calendrier_matchs(calendrier: pd.DataFrame) -> pd.DataFrame:
    requis = {
        "match_id",
        "journee",
        "date",
        "lieu",
        "domicile_club_id",
        "exterieur_club_id",
        "nom_uefa",
        "adversaire_uefa",
    }
    manquantes = requis - set(calendrier.columns)
    if manquantes:
        raise ValueError(
            "Colonnes manquantes dans `calendrier_2026.csv` : "
            + ", ".join(sorted(manquantes))
        )
    domiciles = calendrier.loc[calendrier["lieu"].eq("domicile")].copy()
    matchs = domiciles[
        [
            "match_id",
            "journee",
            "date",
            "domicile_club_id",
            "exterieur_club_id",
            "nom_uefa",
            "adversaire_uefa",
        ]
    ].rename(
        columns={
            "nom_uefa": "domicile_uefa",
            "adversaire_uefa": "exterieur_uefa",
        }
    )
    matchs = matchs.sort_values(["journee", "date", "match_id"]).reset_index(drop=True)
    if len(matchs) != 144 or not matchs["match_id"].is_unique:
        raise ValueError(
            "`calendrier_2026.csv` doit contenir les 144 matchs officiels, chacun une seule fois."
        )
    return matchs


def _elo_par_club(elo: pd.DataFrame) -> dict[int, float]:
    if not {"club_id", "elo_complet"}.issubset(elo.columns):
        raise ValueError(
            "`elo_participants_2026.csv` doit contenir les colonnes `club_id` et `elo_complet`."
        )
    return elo.set_index("club_id")["elo_complet"].astype(float).to_dict()


def _trouver_match_prediction(
    predictions: pd.DataFrame, match: pd.Series
) -> pd.Series:
    col_id = _colonne(predictions, ("match_id", "id_match"))
    if col_id and match["match_id"] in predictions[col_id].values:
        return predictions.loc[predictions[col_id].eq(match["match_id"])].iloc[0]

    col_dom_id = _colonne(
        predictions, ("domicile_club_id", "home_club_id", "home_id")
    )
    col_ext_id = _colonne(
        predictions, ("exterieur_club_id", "away_club_id", "away_id")
    )
    if col_dom_id and col_ext_id:
        trouve = predictions.loc[
            predictions[col_dom_id].eq(match["domicile_club_id"])
            & predictions[col_ext_id].eq(match["exterieur_club_id"])
        ]
        if not trouve.empty:
            return trouve.iloc[0]

    col_dom = _colonne(
        predictions, ("domicile_uefa", "home_uefa", "domicile", "home", "home_team")
    )
    col_ext = _colonne(
        predictions, ("exterieur_uefa", "away_uefa", "exterieur", "away", "away_team")
    )
    if col_dom and col_ext:
        trouve = predictions.loc[
            predictions[col_dom].eq(match["domicile_uefa"])
            & predictions[col_ext].eq(match["exterieur_uefa"])
        ]
        if not trouve.empty:
            return trouve.iloc[0]
    raise ValueError(
        "Impossible de retrouver le match correspondant dans `predictions_phase_ligue_2026.csv` "
        "par `match_id` ou par noms UEFA domicile/extérieur."
    )


def _verifier_probabilites(
    calendrier: pd.DataFrame,
    elo: pd.DataFrame,
    forces: pd.DataFrame,
    predictions: pd.DataFrame,
    modele: Any,
) -> str:
    matchs = _calendrier_matchs(calendrier)
    if matchs.empty:
        raise ValueError("Aucun match à utiliser pour la vérification de démarrage.")
    match = matchs.iloc[0]
    elo_clubs = _elo_par_club(elo)
    ecart_elo = (
        elo_clubs[match["domicile_club_id"]]
        - elo_clubs[match["exterieur_club_id"]]
    )
    entrees = entrees_match(
        _forces_par_club(forces), match["domicile_club_id"], match["exterieur_club_id"]
    )
    probs = probas_match(modele, ecart_elo, domicile=1, **entrees)
    ligne = _trouver_match_prediction(predictions, match)
    colonnes = {
        "Victoire domicile": _colonne(
            predictions,
            (
                "p_victoire_domicile",
                "proba_victoire_domicile",
                "proba_victoire_dom",
                "proba_domicile",
                "proba_dom",
                "proba_1",
                "p_domicile",
                "p_dom",
                "p_home",
                "p_home_win",
                "prob_home_win",
            ),
        ),
        "Nul": _colonne(
            predictions,
            ("p_nul", "proba_nul", "proba_x", "p_x", "p_draw", "prob_draw"),
        ),
        "Victoire extérieur": _colonne(
            predictions,
            (
                "p_victoire_exterieur",
                "proba_victoire_exterieur",
                "proba_victoire_ext",
                "proba_exterieur",
                "proba_ext",
                "proba_2",
                "p_exterieur",
                "p_ext",
                "p_away",
                "p_away_win",
                "prob_away_win",
            ),
        ),
    }
    if any(colonne is None for colonne in colonnes.values()):
        raise ValueError(
            "`predictions_phase_ligue_2026.csv` doit fournir les probabilités de victoire "
            "à domicile, de nul et de victoire à l’extérieur."
        )
    ecarts = {
        issue: abs(probs[issue] - float(ligne[colonne]))
        for issue, colonne in colonnes.items()
        if colonne is not None
    }
    if any(ecart > 0.001 for ecart in ecarts.values()):
        details = ", ".join(f"{issue}: {ecart:.4f}" for issue, ecart in ecarts.items())
        return f"Échec de la vérification des probabilités (écart supérieur à 0,001) — {details}."
    return "Vérification au démarrage réussie : les probabilités correspondent au CSV à 0,001 près."


def _charger_modele_requis() -> Any | None:
    chemin = DOSSIER / FICHIERS["modele"]
    if not chemin.is_file():
        _erreur_fichier("modele")
        return None
    modele = charger_modele(str(chemin))
    cles_requises = {
        "intercept",
        "coef_ecart_100",
        "coef_terrain",
        "coef_nouveau_format",
        "coef_force_ad",
        "coef_valeur_recente",
        "mediane_force_ad",
        "max_buts",
    }
    if not isinstance(modele, dict) or not cles_requises.issubset(modele):
        st.error(
            "`modele_scores.joblib` n’a pas le format exporté par "
            "`analyse/MODEL_POISSON.ipynb` (version 3) : ré-exécutez ce notebook."
        )
        return None
    return modele


def _afficher_page_prediction() -> None:
    st.header("Prédire un match")
    calendrier = _charger_csv_requis("calendrier")
    elo = _charger_csv_requis("elo")
    forces = _charger_csv_requis("forces")
    predictions = _charger_csv_requis("predictions")
    modele = _charger_modele_requis()
    if any(donnee is None for donnee in (calendrier, elo, forces, predictions, modele)):
        return

    try:
        _calendrier_matchs(calendrier)
        elo_clubs = _elo_par_club(elo)
        forces_clubs = _forces_par_club(forces)
        verification = _verifier_probabilites(calendrier, elo, forces, predictions, modele)
    except (AttributeError, KeyError, TypeError, ValueError) as erreur:
        st.error(f"Vérification au démarrage impossible : {erreur}")
        return
    if verification.startswith("Échec"):
        st.error(verification)
        return
    st.success(verification)
    clubs = calendrier[["club_id", "nom_uefa"]].drop_duplicates().sort_values("nom_uefa")
    noms = clubs["nom_uefa"].tolist()
    col_dom, col_ext = st.columns(2)
    domicile = col_dom.selectbox("Club qui reçoit", noms, key="club_domicile")
    choix_exterieurs = [club for club in noms if club != domicile]
    exterieur = col_ext.selectbox("Club qui se déplace", choix_exterieurs, key="club_exterieur")
    terrain = st.radio("Terrain", ("domicile", "neutre"), horizontal=True)

    ids = clubs.set_index("nom_uefa")["club_id"]
    id_dom, id_ext = ids[domicile], ids[exterieur]
    if id_dom not in elo_clubs or id_ext not in elo_clubs:
        st.error("L’Elo complet d’un des clubs est absent de `elo_participants_2026.csv`.")
        return
    if id_dom not in forces_clubs.index or id_ext not in forces_clubs.index:
        st.error("Les forces d’un des clubs sont absentes de `forces_participants_2026.csv`.")
        return
    valeurs = forces_clubs["valeur_recente_eur"]
    st.caption(
        f"Elo — {domicile} : {elo_clubs[id_dom]:.0f} · {exterieur} : {elo_clubs[id_ext]:.0f}  \n"
        f"Valeur des 11 joueurs les plus utilisés — {domicile} : {valeurs[id_dom] / 1e6:.0f} M€ · "
        f"{exterieur} : {valeurs[id_ext] / 1e6:.0f} M€"
    )
    try:
        ecart_elo = elo_clubs[id_dom] - elo_clubs[id_ext]
        terrain_dom = int(terrain == "domicile")
        entrees = entrees_match(forces_clubs, id_dom, id_ext)
        buts_dom, buts_ext = buts_attendus(modele, ecart_elo, domicile=terrain_dom, **entrees)
        probas = probas_match(modele, ecart_elo, domicile=terrain_dom, **entrees)
        scores = matrice_scores(modele, ecart_elo, domicile=terrain_dom, **entrees)
    except (AttributeError, KeyError, TypeError, ValueError) as erreur:
        st.error(f"Le modèle ne peut pas prédire ce match : {erreur}")
        return

    col_dom.metric(f"Buts attendus — {domicile}", f"{buts_dom:.2f}")
    col_ext.metric(f"Buts attendus — {exterieur}", f"{buts_ext:.2f}")

    st.subheader("Probabilités du résultat")
    issues = {
        f"Victoire {domicile}": probas["Victoire domicile"],
        "Match nul": probas["Nul"],
        f"Victoire {exterieur}": probas["Victoire extérieur"],
    }
    # Chaque barre est sous son propre chiffre, dans le même ordre : domicile, nul, extérieur
    for colonne, (issue, proba) in zip(st.columns(3), issues.items()):
        colonne.metric(issue, f"{proba:.0%}")
        colonne.progress(proba)

    st.subheader("Les 5 scores les plus probables")
    top5 = pd.DataFrame(
        [
            {"Score": f"{domicile} {i} – {j} {exterieur}", "Probabilité": proba}
            for i, j, proba in scores_les_plus_probables(scores, 5)
        ]
    )
    st.dataframe(
        top5,
        hide_index=True,
        width="stretch",
        column_config={
            "Probabilité": st.column_config.ProgressColumn(
                "Probabilité",
                format="percent",
                min_value=0.0,
                max_value=float(top5["Probabilité"].max()),
            )
        },
    )
    buts_totaux = np.add.outer(np.arange(scores.shape[0]), np.arange(scores.shape[1]))
    col_buts, col_deux = st.columns(2)
    col_buts.metric("3 buts ou plus dans le match", f"{scores[buts_totaux >= 3].sum():.0%}")
    col_deux.metric("Les deux équipes marquent", f"{scores[1:, 1:].sum():.0%}")
    st.caption(
        f"Ces 5 scores ne couvrent ensemble que {top5['Probabilité'].sum():.0%} des cas : "
        "le football reste très incertain."
    )

    with st.expander("Probabilité de chaque score (0–0 à 5–5)"):
        matrice = pd.DataFrame(
            scores[:6, :6],
            index=[f"{buts} but(s) {domicile}" for buts in range(6)],
            columns=[f"{buts} but(s) {exterieur}" for buts in range(6)],
        )
        st.dataframe(
            matrice.style.format("{:.1%}").background_gradient(cmap="YlGn"),
            width="stretch",
        )


def _afficher_page_calendrier() -> None:
    st.header("Phase de ligue 2026-27")
    calendrier = _charger_csv_requis("calendrier")
    if calendrier is None:
        return
    try:
        matchs = _calendrier_matchs(calendrier)
    except ValueError as erreur:
        st.error(str(erreur))
        return
    journees = sorted(matchs["journee"].dropna().unique().tolist())
    clubs = sorted(set(matchs["domicile_uefa"]) | set(matchs["exterieur_uefa"]))
    choix_journee = st.selectbox("Journée", ["Toutes"] + journees)
    choix_club = st.selectbox("Club", ["Tous"] + clubs)
    affichage = matchs.copy()
    if choix_journee != "Toutes":
        affichage = affichage.loc[affichage["journee"].eq(choix_journee)]
    if choix_club != "Tous":
        affichage = affichage.loc[
            affichage["domicile_uefa"].eq(choix_club)
            | affichage["exterieur_uefa"].eq(choix_club)
        ]
    st.dataframe(
        affichage[
            ["journee", "date", "domicile_uefa", "exterieur_uefa"]
        ].reset_index(drop=True),
        hide_index=True,
        width="stretch",
    )


COLONNES_SIMULATION = {
    "nom_uefa": "Club",
    "points_moyens": "Points moyens",
    "pct_top8": "Top 8 (%)",
    "pct_huitiemes": "Huitièmes (%)",
    "pct_quarts": "Quarts (%)",
    "pct_demies": "Demies (%)",
    "pct_finale": "Finale (%)",
    "pct_titre": "Titre (%)",
}


def _afficher_page_simulation() -> None:
    st.header("Simulation de la saison")
    chemin = DOSSIER / FICHIERS["simulation"]
    if not chemin.is_file():
        st.info(
            "La simulation n'est pas encore disponible : exécutez `analyse/SIMULATION_2026.ipynb`."
        )
        return
    resultats = lire_csv(FICHIERS["simulation"])
    if not set(COLONNES_SIMULATION).issubset(resultats.columns):
        colonne_titre = _colonne(
            resultats,
            ("proba_titre", "probabilite_titre", "probabilité_titre", "titre", "p_titre"),
        )
        if colonne_titre is None:
            st.error(
                "`resultats_simulation_2026.csv` doit contenir une colonne de probabilité de titre."
            )
            return
        st.dataframe(
            resultats.sort_values(colonne_titre, ascending=False),
            hide_index=True,
            width="stretch",
        )
        return

    tableau = (
        resultats[list(COLONNES_SIMULATION)]
        .rename(columns=COLONNES_SIMULATION)
        .sort_values("Titre (%)", ascending=False)
        .reset_index(drop=True)
    )
    favori = tableau.iloc[0]
    nb = f"{int(resultats['nb_simulations'].iloc[0]):,}".replace(",", " ") if "nb_simulations" in resultats else "plusieurs milliers de"
    titre = f"{favori['Titre (%)']:.1f}".replace(".", ",")
    finale = f"{favori['Finale (%)']:.1f}".replace(".", ",")
    st.markdown(
        f"Sur **{nb}** saisons simulées, le favori est **{favori['Club']}** : "
        f"titre dans **{titre} %** des cas, finale dans {finale} %."
    )

    st.subheader("Les 10 favoris au titre")
    st.bar_chart(
        tableau.head(10).set_index("Club")["Titre (%)"],
        horizontal=True,
        x_label="Probabilité de titre (%)",
        y_label="",
        sort="-Titre (%)",
    )

    st.subheader("Toutes les équipes")
    pourcentages = [colonne for colonne in tableau.columns if colonne.endswith("(%)")]
    st.dataframe(
        tableau,
        hide_index=True,
        width="stretch",
        column_config={
            "Points moyens": st.column_config.NumberColumn(format="%.1f"),
            **{
                colonne: st.column_config.ProgressColumn(
                    colonne, format="%.1f", min_value=0.0, max_value=100.0
                )
                for colonne in pourcentages
            },
        },
    )
    st.caption(
        "Chaque saison est jouée match par match avec le modèle de scores, "
        "et l’Elo des clubs est mis à jour après chaque rencontre."
    )


def main() -> None:
    st.set_page_config(page_title="Ligue des champions 2026-27", page_icon="⚽", layout="wide")
    st.title("Ligue des champions — saison 2026-27")
    st.info(
        "Estimation statistique : le modèle connaît l'Elo des clubs, leurs forces d'attaque et de "
        "défense, la valeur de leurs joueurs récents, le terrain et le format. "
        "Il ignore les blessures et les compositions du jour."
    )
    page = st.sidebar.radio(
        "Navigation",
        ("Prédire un match", "Phase de ligue 2026-27", "Simulation"),
    )

    if page == "Prédire un match":
        _afficher_page_prediction()
    elif page == "Phase de ligue 2026-27":
        _afficher_page_calendrier()
    else:
        _afficher_page_simulation()


if __name__ == "__main__":
    main()
