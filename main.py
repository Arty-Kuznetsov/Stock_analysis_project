import pandas as pd
import streamlit as st
import yfinance
import talib
import plotly.graph_objects as go
import re
import feedparser
import csv
from urllib.request import urlopen, Request
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs

@st.cache
def load_data():
    components = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return components.drop('SEC filings', axis=1).set_index('Symbol')



def load_quotes(asset):
    return yfinance.download(asset)


def main():

   # print(os.getcwd())

    components = load_data()
    title = st.empty()
    st.sidebar.title("Options")

    def label(symbol):
        a = components.loc[symbol]
        return symbol + ' - ' + a.Security

    if st.sidebar.checkbox('View companies list'):
        dat = st.dataframe(components[['Security',
                                 'GICS Sector',
                                 'Date first added',
                                 'Founded']])

        st.dataframe(components[[
                                 'GICS Sector',
                                 ]].sort_values('GICS Sector'))


        #mpl.plot(dat)
        #st.dataframe(components[['Security', 'GICS Sector']].groupby('GICS Sector')).value_counts()
        #dat['Col'] = dat
        #mpl.pie(['Security'], labels=['GICS Sector'])

    st.sidebar.subheader('Select asset')
    asset = st.sidebar.selectbox('Click below to select a new asset',
                                 components.index.sort_values(), index=3,
                                 format_func=label)
    title.title(components.loc[asset].Security)
    #print(asset)
    if st.sidebar.checkbox('View company info', True):
        st.table(components.loc[asset])
    data0 = load_quotes(asset)
    data = data0.copy().dropna()
    data.index.name = None
    data['Date'] = data.index
    section = st.sidebar.slider('Number of quotes', min_value=30,
                        max_value=min([2000, data.shape[0]]),
                        value=500,  step=10)

    data2 = data[-section:]['Adj Close'].to_frame('Adj Close')


    st.subheader('Chart')

    st.line_chart(data2, width=1)

    print(data.head())


    if st.sidebar.checkbox('Technical analysis'):

        if st.sidebar.checkbox('SMA'):
            period= st.sidebar.slider('SMA period', min_value=5, max_value=500,
                                 value=20,  step=1)
            data[f'SMA {period}'] = data['Adj Close'].rolling(period ).mean()
            data2[f'SMA {period}'] = data[f'SMA {period}']

        if st.sidebar.checkbox('SMA2'):
            period2= st.sidebar.slider('SMA2 period', min_value=5, max_value=500,
                                 value=100,  step=1)
            data[f'SMA2 {period2}'] = data['Adj Close'].rolling(period2).mean()
            data2[f'SMA2 {period2}'] = data[f'SMA2 {period2}']

        fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=True, height=450)

        st.plotly_chart(fig)

       # mpf.plot(data2, type='candle')

        #if st.sidebar.checkbox('View stadistic'):
        #    st.subheader('Stadistic')
        #    st.table(data2.describe())

        if st.sidebar.checkbox('View quotes'):
            st.subheader(f'{asset} historical data')
            st.write(data)

        if st.sidebar.checkbox('RSI'):
            data2['RSI'] = talib.RSI(data['Adj Close'], timeperiod=14)
            st.header(f"Relative Strength Index\n")
            st.line_chart(data2['RSI'])
            st.subheader(f"RSI today:")
            st.write(data2['RSI'].values[-1])

        if st.sidebar.checkbox('Volume'):
            st.line_chart(data['Volume'])

        if st.sidebar.checkbox('Bollinger bands'):
            data2['upper_band'], data2['middle_band'], data2['lower_band'] = talib.BBANDS(data2['Adj Close'], timeperiod=20)
            st.header(f"Bollinger Bands")
            st.line_chart(data2[['Adj Close', 'upper_band', 'middle_band', 'lower_band']])

        if st.sidebar.checkbox('MACD'):
            data2['macd'], data2['macdsignal'], data2['macdhist'] = talib.MACD(data2['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            st.line_chart(data2[['macd', 'macdsignal']])


    st.sidebar.subheader('News')
    if st.sidebar.checkbox('Politics'):
        newsurls = {'Kommersant': 'https://www.kommersant.ru/RSS/news.xml',
                    'Lenta.ru': 'https://lenta.ru/rss/',
                    'Vesti': 'https://www.vesti.ru/vesti.rss',
                    'Bloomberg politics':'https://feedfry.com/rss/11ecce9cce613d0890eefe4f58c582a7',
                    'Gurutrade':'https://www.gurutrade.ru/news/economic/rss/'}
        # пример словаря RSS-лент
        # русскоязычных источников

        f_all_news = "C:\\Users\\artyg\\PycharmProjects\\pythonProjectnorm\\allnews.csv"
        f_certain_news = "C:\\Users\\artyg\\PycharmProjects\\pythonProjectnorm\\certainnews.csv"

        vector1 = components.loc[asset].Security # пример таргетов

        vector2 = ''

        def parseRSS(rss_url):  # функция получает линк на рсс ленту, возвращает распаршенную ленту с помощью feedpaeser
            return feedparser.parse(rss_url)

        def getHeadlines(rss_url):  # функция для получения заголовков новости
            headlines = []
            feed = parseRSS(rss_url)
            for newsitem in feed['items']:
                headlines.append(newsitem['title'])
            return headlines

        def getDescriptions(rss_url):  # функция для получения описания новости
            descriptions = []
            feed = parseRSS(rss_url)
            for newsitem in feed['items']:
                descriptions.append(newsitem['description'])
            return descriptions

        def getLinks(rss_url):  # функция для получения ссылки на источник новости
            links = []
            feed = parseRSS(rss_url)
            for newsitem in feed['items']:
                links.append(newsitem['link'])
            return links

        def getDates(rss_url):  # функция для получения даты публикации новости
            dates = []
            feed = parseRSS(rss_url)
            for newsitem in feed['items']:
                dates.append(newsitem['published'])
            return dates

        allheadlines = []
        alldescriptions = []
        alllinks = []
        alldates = []
        # Прогоняем нашии URL и добавляем их в наши пустые списки
        for key, url in newsurls.items():
            allheadlines.extend(getHeadlines(url))

        for key, url in newsurls.items():
            alldescriptions.extend(getDescriptions(url))

        for key, url in newsurls.items():
            alllinks.extend(getLinks(url))

        for key, url in newsurls.items():
            alldates.extend(getDates(url))
        pd.set_option('display.max_colwidth', -1)
        def write_all_news(all_news_filepath):
            # функция для записи всех новостей в .csv, возвращает нам этот датасет
            header = ['Title', 'Description', 'Links', 'Publication Date']

            with open(all_news_filepath, 'w', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')

                writer.writerow(i for i in header)

                for a, b, c, d in zip(allheadlines, alldescriptions,
                                      alllinks, alldates):
                    writer.writerow((a, b, c, d))

                df = pd.read_csv(all_news_filepath)

            return df


        def looking_for_certain_news(all_news_filepath, certain_news_filepath, target1, target2):
            # функция для поиска, а затем записи
            # определенных новостей по таргетам,
            # затем возвращает этот датасет
            df = pd.read_csv(all_news_filepath)

            result = df.apply(lambda x: x.str.contains(target1, na=False,
                                                       flags=re.IGNORECASE, regex=True)).any(axis=1)
            result2 = df.apply(lambda x: x.str.contains(target2, na=False,
                                                        flags=re.IGNORECASE, regex=True)).any(axis=1)
            new_df = df[result & result2]

            new_df.to_csv(certain_news_filepath
                          , sep='\t', encoding='utf-8-sig')

            return new_df

            # новости по векторам

        write_all_news(f_all_news)
        st.write(looking_for_certain_news(f_all_news, f_certain_news, vector1,vector2),unsafe_allow_html=True)

    if st.sidebar.checkbox('Stocktwits'):
        symbol = asset

        r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json")

        data3 = r.json()


        for message in data3['messages']:
            st.image(message['user']['avatar_url'])
            st.write(message['user']['username'])
            st.write(message['created_at'])
            st.write(message['body'])




    st.sidebar.subheader('Analysis')

    analyzer = SentimentIntensityAnalyzer()
    if st.sidebar.checkbox('Sentiments analysis(Finviz)'):
        st.subheader('Sentiment analysis (Finviz)')
        finviz_url = 'https://finviz.com/quote.ashx?t='
        tickers = asset
        news_tables = {}

        url = finviz_url + tickers

        req = Request(url=url, headers={'user-agent': 'my-app'})
        response = urlopen(req)

        html = bs(response, features='html.parser')
        news_table = html.find(id='news-table')
        news_tables[tickers] = news_table

        parsed_data = []

        for ticker, news_table in news_tables.items():

            for row in news_table.findAll('tr'):

                title = row.a.text
                date_data = row.td.text.split(' ')

                if len(date_data) == 1:
                    time = date_data[0]
                else:
                    date = date_data[0]
                    time = date_data[1]

                parsed_data.append([ticker, date, time, title])

        df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

        f = lambda title: analyzer.polarity_scores(title)['compound']
        df['compound'] = df['title'].apply(f)
        df['date'] = pd.to_datetime(df.date).dt.date

        #plt.figure(figsize=(10, 8))
        mean_df = df.groupby(['ticker', 'date']).mean().unstack()
        mean_df = mean_df.xs('compound', axis="columns")
        st.write(mean_df.T)
        #mean_df.T
        st.bar_chart(mean_df.T[asset])

 ############################################################################################

    if st.sidebar.checkbox('Sentiments analysis(NewsAPI)'):


        data['2_SMA'] = data['Close'].rolling(window=14).mean()
        data['5_SMA'] = data['Close'].rolling(window=25).mean()

        data_amd = data[data['5_SMA'].notna()]
        data_amd = data_amd.iloc[-section:, :]
        # SMA trade calls
        Trade_Buy = []
        Trade_Sell = []
        for i in range(len(data_amd) - 1):
            if ((data_amd['2_SMA'].values[i] < data_amd['5_SMA'].values[i]) & (
                    data_amd['2_SMA'].values[i + 1] > data_amd['5_SMA'].values[i + 1])):
                #print("Trade Call for {row} is Buy.".format(row=data_amd.index[i].date()))
                Trade_Buy.append(i)
            elif ((data_amd['2_SMA'].values[i] > data_amd['5_SMA'].values[i]) & (
                    data_amd['2_SMA'].values[i + 1] < data_amd['5_SMA'].values[i + 1])):
                #print("Trade Call for {row} is Sell.".format(row=data_amd.index[i].date()))
                Trade_Sell.append(i)

        # Define the endpoint
        url = 'https://newsapi.org/v2/everything?'

        # Specify the query and number of returns
        parameters = {
            'q': asset,  # query phrase
            'sortBy': 'popularity',  # articles from popular sources and publishers come first
            'pageSize': 100,  # maximum is 100 for developer version
            'apiKey': '18fb02217a0f46baaf14b9dc28d6af43',  # your own API key
        }

        # Make the request
        response = requests.get(url, params=parameters)

        # Convert the response to JSON format and store it in dataframe
        data = pd.DataFrame(response.json())

        news_df = pd.concat([data['articles'].apply(pd.Series)], axis=1)

        final_news = news_df.loc[:, ['publishedAt', 'title']]
        final_news['publishedAt'] = pd.to_datetime(final_news['publishedAt'])
        final_news.sort_values(by='publishedAt', inplace=True)

        # Import BDay to determine business day's dates
        from pandas.tseries.offsets import BDay

        # to get the business day for which particular news headline should be used to make trade calls
        def get_trade_open(date):
            curr_date_open = pd.to_datetime(date).floor('d').replace(hour=13, minute=30) - BDay(0)
            curr_date_close = pd.to_datetime(date).floor('d').replace(hour=20, minute=0) - BDay(0)

            prev_date_close = (curr_date_open - BDay()).replace(hour=20, minute=0)
            next_date_open = (curr_date_close + BDay()).replace(hour=13, minute=30)

            if ((pd.to_datetime(date) >= prev_date_close) & (pd.to_datetime(date) < curr_date_open)):
                return curr_date_open
            elif ((pd.to_datetime(date) >= curr_date_close) & (pd.to_datetime(date) < next_date_open)):
                return next_date_open
            else:
                return None

        # Apply the above function to get the trading time for each news headline
        final_news["trading_time"] = final_news["publishedAt"].apply(get_trade_open)

        final_news = final_news[pd.notnull(final_news['trading_time'])]
        final_news['Date'] = pd.to_datetime(pd.to_datetime(final_news['trading_time']).dt.date)
        cs = []
        for row in range(len(final_news)):
            cs.append(analyzer.polarity_scores(final_news['title'].iloc[row])['compound'])

        final_news['compound_vader_score'] = cs
        final_news = final_news[(final_news[['compound_vader_score']] != 0).all(axis=1)].reset_index(drop=True)
        final_news['Date'] = final_news['Date']
        final_news['Date2'] = final_news['Date'].dt.date
        ### Plot data and Bar

        st.subheader('Sentiment analysis (NewsAPI)')
        st.dataframe(final_news)
        st.bar_chart(final_news['compound_vader_score'])
         ################################
        unique_dates = final_news['Date'].unique()
        grouped_dates = final_news.groupby(['Date'])
        keys_dates = list(grouped_dates.groups.keys())

        max_cs = []
        min_cs = []

        for key in grouped_dates.groups.keys():
            data = grouped_dates.get_group(key)
            if data["compound_vader_score"].max() > 0:
                max_cs.append(data["compound_vader_score"].max())
            elif data["compound_vader_score"].max() < 0:
                max_cs.append(0)

            if data["compound_vader_score"].min() < 0:
                min_cs.append(data["compound_vader_score"].min())
            elif data["compound_vader_score"].min() > 0:
                min_cs.append(0)

        extreme_scores_dict = {'Date': keys_dates, 'max_scores': max_cs, 'min_scores': min_cs}
        extreme_scores_df = pd.DataFrame(extreme_scores_dict)
        final_scores = []
        for i in range(len(extreme_scores_df)):
            final_scores.append(extreme_scores_df['max_scores'].values[i] + extreme_scores_df['min_scores'].values[i])

        extreme_scores_df['final_scores'] = final_scores


        # VADER trade calls - with threshold

        vader_Buy = []
        vader_Sell = []
        for i in range(len(extreme_scores_df)):
            if extreme_scores_df['final_scores'].values[i] > 0.20:
               # print("Trade Call for {row} is Buy.".format(row=extreme_scores_df['Date'].iloc[i].date()))
                vader_Buy.append(extreme_scores_df['Date'].iloc[i].date())
            elif extreme_scores_df['final_scores'].values[i] < -0.20:
               # print("Trade Call for {row} is Sell.".format(row=extreme_scores_df['Date'].iloc[i].date()))
                vader_Sell.append(extreme_scores_df['Date'].iloc[i].date())
        vader_buy = []
        for i in range(len(data_amd)):
            if data_amd.index[i].date() in vader_Buy:
                vader_buy.append(i)

        vader_sell = []

        for i in range(len(data_amd)):
            if data_amd.index[i].date() in vader_Sell:
                vader_sell.append(i)

        # prioritising SMA signals
        final_buy = list(set(Trade_Buy + vader_buy) - set(Trade_Sell))
        final_sell = list(set(Trade_Sell + vader_sell) - set(Trade_Buy))

        fig = plt.figure(figsize=(20, 10), dpi=80)
        plt.plot(data_amd.index, data_amd['2_SMA'], color='blue')
        plt.plot(data_amd.index, data_amd['5_SMA'], color='orange')
        plt.plot(data_amd.index, data_amd['Close'], '-^', markevery=final_buy, ms=15, color='green')
        plt.plot(data_amd.index, data_amd['Close'], '-v', markevery=final_sell, ms=15, color='red')
        plt.plot(data_amd.index, data_amd['Close'])
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price in Dollars', fontsize=14)
        plt.xticks(rotation='60', fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('Trade Calls - SMA with Vader', fontsize=16)
        plt.legend(['2_SMA', '5_SMA', 'Buy', 'Sell', 'Close'])
        plt.grid()
        plt.show()
        st.pyplot(fig)

    if st.sidebar.checkbox('Sentiment analysis(Twits)'):
        st.subheader('Sentiment analysis(Twits)')
        symbol = asset

        r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json")

        data3 = r.json()
        data_twit = pd.DataFrame(data3['messages'])
        fun = lambda body: analyzer.polarity_scores(body)['compound']
        data_twit['compound'] = data_twit['body'].apply(fun)
        data_twit0 = data_twit.loc[:, ['created_at','body', 'compound']]
        data_twit = data_twit.loc[:,['created_at' , 'compound']]
        data_twit['created_at'] = pd.to_datetime(data_twit['created_at'])
        data_twit.sort_values(by='created_at', inplace=True)
        st.write(data_twit0)
       #st.write(data_twit)
        st.bar_chart(data_twit['compound'])




if __name__ == '__main__':
    main()