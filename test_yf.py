import yfinance as yf
import subprocess
import sys

def test_yahoo_connection():
    """ Vérifie si Yahoo Finance est accessible. """
    try:
        test_data = yf.download("BTC-USD", period="1d", interval="1h")
        if test_data.empty:
            print("⚠️ Connexion établie, mais aucune donnée reçue.")
        else:
            print("✅ Connexion à Yahoo Finance réussie.")
        return not test_data.empty
    except Exception as e:
        print(f"❌ Erreur de connexion à Yahoo Finance : {e}")
        return False

def test_btc_download():
    """ Télécharge BTC/USD avec différents paramètres et vérifie la réponse. """
    settings = [
        ("5d", "5m"),
        ("1d", "5m"),
        ("5d", "1h")
    ]
    for period, interval in settings:
        print(f"\n🔄 Test téléchargement BTC/USD - Période: {period}, Intervalle: {interval}")
        try:
            data = yf.download("BTC-USD", period=period, interval=interval)
            if data.empty:
                print(f"⚠️ Aucune donnée reçue pour {period} {interval}.")
            else:
                print(f"✅ Données téléchargées avec succès ({len(data)} lignes).")
                print(data.tail())
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement: {e}")

def test_other_asset():
    """ Teste avec un autre actif (AAPL) pour vérifier si le problème vient de BTC. """
    print("\n🔄 Test téléchargement AAPL (Apple)")
    try:
        data = yf.download("AAPL", period="5d", interval="5m")
        if data.empty:
            print("⚠️ Aucune donnée reçue pour AAPL.")
        else:
            print(f"✅ Données téléchargées pour AAPL ({len(data)} lignes).")
            print(data.tail())
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de AAPL : {e}")

def update_yfinance():
    """ Met à jour yfinance si nécessaire. """
    print("\n🔄 Mise à jour de yfinance...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yfinance"])
        print("✅ yfinance mis à jour avec succès.")
    except Exception as e:
        print(f"❌ Échec de la mise à jour de yfinance : {e}")

if __name__ == "__main__":
    print("🔍 Début des tests...\n")
    
    if not test_yahoo_connection():
        print("⛔ Fin du script : Yahoo Finance semble inaccessible.")
    else:
        test_btc_download()
        test_other_asset()
        update_yfinance()
    
    print("\n✅ Tests terminés.")
