import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta


end_date = datetime.datetime.now()
print(end_date)
# Obliczenie daty i godziny sprzed 24 godzin
start_date = end_date - timedelta(days=30)
print(start_date)

# Formatowanie dat do postaci akceptowanej przez Yahoo Finance
start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
# Funkcja do pobierania danych dotyczących akcji, rysowania wykresu i zapisywania do pliku
def download_stock_data(ticker):
    # Pobierz dane dotyczące akcji z ostatniego roku
    data = yf.download(ticker, start=start_date, end=end_date,interval='1d')
    print(data)
    # Narysuj wykres ceny zamknięcia
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.title(f'{ticker} Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    
    # Zapisz wykres do pliku
    filename = f"{ticker}_chart.jpg"
    plt.savefig(filename)
    print(f"Wykres dla {ticker} został zapisany do pliku {filename}")

# Lista tickerów (symboli) akcji, dla których chcemy pobrać i narysować wykresy
#gainers
#tickers = ['ODD', 'ZKH', 'MBLY', 'NEMCL', 'NRSCF', 'JAZZ', 'RNW', 'SRPT', 'ARCB', 'FOUR','CRDO','TTC','TTC','CARG','TMDX']
#loosers
#tickers=['GME','IOT',"MTN","HMY","CDE","PI","MPNGY","NVAX","LNVGY","UPST","PAAS","HL","EGO","AEM","AU"]
#test
tickers=['BTC-USD']
# Pętla po tickerach
for ticker in tickers:
    download_stock_data(ticker)
