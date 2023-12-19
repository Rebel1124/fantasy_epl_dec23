# Initial imports
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import numpy as np
import datetime as dt
from pathlib import Path
import math
import os
from scipy.stats import poisson,skellam
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import date, timedelta
import streamlit as st
from PIL import Image
import time
import warnings


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

warnings.filterwarnings("ignore")

# App Inputs for user selection
games=6
targTeam = 'Man United'
teama = 'Man United'
teamb = 'Liverpool'
graph_value= 'Points'


# User selection menus
teams = ['Arsenal',
         'Aston Villa',
         'Bournemouth',
         'Brentford',
         'Brighton',
         'Chelsea',
         'Crystal Palace',
         'Everton',
         'Fulham',
         'Leeds',
         'Leicester',
         'Liverpool',
         'Man City',
         'Man United',
         'Newcastle',
         "Nott'm Forest",
         'Southampton',
         'Tottenham',
         'West Ham',
         'Wolves']



graph_options = ['Won', 'Draw', 'Lost', 'Goals For', 'Goals Against', 'Shots For', 'Shots Against', 'T-Shots For','T-Shots Against', 'Points']


image = Image.open('football_logo/Premier.png')

colb, colc = st.columns([1, 4.5])
colb.image(image, use_column_width=True)

colc.markdown("<h1 style='text-align: center; color: #551A8B; padding-left: 0px; font-size: 50px'>English Premier League</h1>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: 	#008080; padding-left: 0px; font-size: 50px'>Football Prediction App</h2>", unsafe_allow_html=True)

st.markdown(" ")
    

####### Functions to import data and plot graphs and tables ###################################################################################

@st.cache(allow_output_mutation=True)
def data(file):
    df = pd.read_csv(file)
    return df

seasons = data("Processed_Data/seasons.csv")
team_merge = data("Processed_Data/team_merge.csv")
merged_stats = data("Processed_Data/merged_stats.csv")
elements = data("FPL_Data/elements.csv")
history = data("FPL_Data/all_history_df_current.csv")



#@st.cache(allow_output_mutation=True)
def changeName(position):
    if(position == 'Goalkeeper'):
        return 'a'
    elif(position == 'Defender'):
        return 'b'
    elif(position == 'Midfielder'):
        return 'c'
    else:
        return 'd'
    
    #return team


elements['positionName'] = elements['position'].apply(changeName)

## Simulate match using Poisson Distribution Model
@st.cache(allow_output_mutation=True)
def simulate_match(home_goals_avg, away_goals_avg, max_goals=10):
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))


@st.cache(allow_output_mutation=True)
def matchProb(homePower, awayPower):
    matrix = simulate_match(homePower,awayPower)
    probHomeWin = np.sum(np.tril(matrix, -1))
    probDraw = np.sum(np.diag(matrix))
    probAwayWin = np.sum(np.triu(matrix, 1))
    
    results = [probHomeWin, probDraw, probAwayWin]
    return results



@st.cache(allow_output_mutation=True)
def tables(team, previousGames, targetTeam):
    
    team = team[-previousGames:]
    
    results =[]
    colours = []
    
    for i in range(0, previousGames):
        if((team['HomeTeam'].iloc[i] == targetTeam) & (team['FTR'].iloc[i] == 'H')):
            results.append('Won')
            colours.append('#98FB98')
        elif((team['AwayTeam'].iloc[i] == targetTeam) & (team['FTR'].iloc[i] == 'A')):
            results.append('Won')
            colours.append('#98FB98')
        elif((team['FTR'].iloc[i] == 'D')):
            results.append('Draw')
            colours.append('#E3CF57')
        else:
            results.append('Lost')
            colours.append('#F08080')

    
    
    head = ['<b>Date<b>', '<b>HomeTeam<b>', '<b>AwayTeam<b>', '<b>FTHG<b>', '<b>FTAG<b>', '<b>HS<b>','<b>AS<b>','<b>HST<b>',
            '<b>AST<b>','<b>FTR<b>', '<b>Result<b>']
    labels = []
    date = team['gameDate'].tolist()
    hTeam = team['HomeTeam'].tolist()
    aTeam = team['AwayTeam'].tolist()
    hg = team['FTHG'].tolist()
    ag = team['FTAG'].tolist()
    hs = team['HS'].tolist()
    ash = team['AS'].tolist()
    hst = team['HST'].tolist()
    ast = team['AST'].tolist()
    res = team['FTR'].tolist()
    
    fig = go.Figure(data=[go.Table(
        columnorder = [1,2,3,4,5,6,7,8,9,10,11],
        columnwidth = [60,80,80,35,35,35,35,35,35,35,35],
        header=dict(values=head,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[date, hTeam, aTeam, hg, ag, hs, ash, hst, ast, res, results],
                   fill_color=[['lavender'], ['lavender'],['lavender'],['lavender'],['lavender'],['lavender'],
                               ['lavender'],['lavender'],['lavender'],['lavender'], colours],
                   align='left'))
    ])
    

    fig.update_layout(height=(25*games), width=700, margin=dict(l=0, r=0, b=0,t=0))
    
    return fig




@st.cache(allow_output_mutation=True)
def teamTables(teamiFPL):
    
    teamiFPL['strength'] = teamiFPL.apply(lambda x: round((x['strength']),2), axis=1)
       
    name = teamiFPL['web_name'].tolist()
    position = teamiFPL['position'].tolist()
    status = teamiFPL['statusFull'].tolist()
    strength = teamiFPL['strength'].tolist()
    category = teamiFPL['str-cat'].tolist()
    
    
    head = ['Name', 'Pos', 'Status', 'Score', 'Cat']
    
    count = len(name)
    
    fig = go.Figure(data=[go.Table(
        
        columnorder = [1,2,3,4,5],
        columnwidth = [58,52,35, 35, 35],
        
        header=dict(values=head,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[name, position, status, strength, category],
                   fill_color='lavender',
                   align='left'))
    ])   

    fig.update_layout(height=(count*25), width=300, margin=dict(l=0, r=0, b=0,t=0))
    
    return fig



@st.cache(allow_output_mutation=True)
def playerTables(filter):
    
    
    name = filter['web_name'].tolist()
    position = filter['position'].tolist()
    cost = filter['now_cost'].tolist()
    sel = filter['selected_by_percent'].tolist()
    minutes = filter['minutes'].tolist()
    goals = filter['goals_scored'].tolist()
    assists = filter['assists'].tolist()
    saves = filter['saves'].tolist()
    form = filter['form'].tolist()
    pts = filter['total_points'].tolist()
    bps = filter['bps'].tolist()
    ict_index = filter['ict_index'].tolist()

    head = ['Name', 'Pos', 'Cost', 'Pick', 'Min', 'Goals', 'Assists', 'Saves', 'Form', 'Points', 'Bonus', 'ICT']
    
    count = len(name)
    
    fig = go.Figure(data=[go.Table(
        
        columnorder = [1,2,3,4,5,6,7,8,9,10,11,12],
        columnwidth = [70,50,30,30,30,30,30,30,30,30,30,30],
        
        header=dict(values=head,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[name, position, cost, sel, minutes, goals, assists, saves, form, pts, bps, ict_index],
                   fill_color='lavender',
                   align='left'))
    ])   

    fig.update_layout(height=(count*22), width=710, margin=dict(l=0, r=0, b=0,t=0))
    
    return fig


@st.cache(allow_output_mutation=True)
def playerStatistics(playerGames, games):
    
    playerGames = playerGames[-games:]
       
    
    date = playerGames['Date'].tolist()
    opponent = playerGames['opponent'].tolist()
    minutes = playerGames['minutes'].tolist()
    goals= playerGames['goals_scored'].tolist()
    assists = playerGames['assists'].tolist()
    saves = playerGames['saves'].tolist()
    pts = playerGames['total_points'].tolist()
    bps = playerGames['bps'].tolist()
    inf = playerGames['influence'].tolist()
    cre = playerGames['creativity'].tolist()
    thr = playerGames['threat'].tolist()
    ict_index = playerGames['ict_index'].tolist()
    
    
    head = ['Date', 'Opp', 'Min', 'Goals', 'Assists', 'Saves', 'Points', 'Bonus', 'Influence', 'Create', 'Threat', 'ICT']
    
    fig22 = go.Figure(data=[go.Table(
        
        columnorder = [1,2,3,4,5,6,7,8,9,10,11,12],
        columnwidth = [55,55,20,25,25,25,25,25,30,25,25,25],
        
        header=dict(values=head,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[date, opponent, minutes, goals, assists, saves, pts, bps, inf, cre, thr, ict_index],
                   fill_color='lavender',
                   align='left'))
    ])   

    fig22.update_layout(height=(games*25), width=710, margin=dict(l=0, r=0, b=0,t=0))
    
    return fig22



@st.cache(allow_output_mutation=True)
def stats(homeTeam, target, games, graph, num):
    
    homeTeam = homeTeam[-games:]

    homewon = 0
    homedraw = 0
    homelost = 0
    homegoalsfor = 0
    homegoalsagainst = 0
    
    homeshotsfor = 0
    homeshotsagainst = 0
    homeshotsTargetfor = 0
    homeshotsTargetagainst = 0
    
    homepoints = 0
    
    awaywon = 0
    awaydraw = 0
    awaylost = 0
    awaygoalsfor = 0
    awaygoalsagainst = 0
    
    awayshotsfor = 0
    awayshotsagainst = 0
    awayshotsTargetfor = 0
    awayshotsTargetagainst = 0
    
    awaypoints = 0
    
    lenth = homeTeam.shape[0]
    
    for i in range(0, lenth):
        if (homeTeam['HomeTeam'].iloc[i] == target):
            
            homegoalsfor += homeTeam['FTHG'].iloc[i]
            homegoalsagainst += homeTeam['FTAG'].iloc[i]
            
            homeshotsfor += homeTeam['HS'].iloc[i]
            homeshotsagainst += homeTeam['AS'].iloc[i]
            homeshotsTargetfor += homeTeam['HST'].iloc[i]
            homeshotsTargetagainst += homeTeam['AST'].iloc[i]
            
            
            
            if(homeTeam['FTR'].iloc[i] == 'H'):
                homewon += 1
                homepoints += 3
            elif(homeTeam['FTR'].iloc[i]=='D'):
                homedraw += 1
                homepoints += 1
            else:
                homelost += 1
                homepoints += 0
                
        else:
            awaygoalsfor += homeTeam['FTAG'].iloc[i]
            awaygoalsagainst += homeTeam['FTHG'].iloc[i]
            
            awayshotsfor += homeTeam['AS'].iloc[i]
            awayshotsagainst += homeTeam['HS'].iloc[i]
            awayshotsTargetfor += homeTeam['AST'].iloc[i]
            awayshotsTargetagainst += homeTeam['HST'].iloc[i]
            
            
            
            if(homeTeam['FTR'].iloc[i] == 'A'):
                awaywon += 1
                awaypoints += 3
            elif(homeTeam['FTR'].iloc[i]=='D'):
                awaydraw += 1
                awaypoints += 1
            else:
                awaylost += 1
                awaypoints += 0
                
            
    Won = homewon + awaywon
    Draw = homedraw + awaydraw
    Lost = homelost + awaylost
    GoalsFor = homegoalsfor + awaygoalsfor
    GoalsAgainst = homegoalsagainst + awaygoalsagainst
    
    ShotsFor = homeshotsfor + awayshotsfor
    ShotsAgainst = homeshotsagainst + awayshotsagainst
    targetShotsFor = homeshotsTargetfor + awayshotsTargetfor
    targetShotsAgainst = homeshotsTargetagainst + awayshotsTargetagainst
    
    
    Points = homepoints + awaypoints


    home = [homewon, homedraw, homelost, homegoalsfor, homegoalsagainst, homeshotsfor, homeshotsagainst,
            homeshotsTargetfor, homeshotsTargetagainst, homepoints]
    away = [awaywon, awaydraw, awaylost, awaygoalsfor, awaygoalsagainst, awayshotsfor, awayshotsagainst,
            awayshotsTargetfor, awayshotsTargetagainst, awaypoints]
    total = [Won, Draw, Lost, GoalsFor, GoalsAgainst, ShotsFor, ShotsAgainst, targetShotsFor, targetShotsAgainst, Points]
    
    
    head = ['<b>Statistic<b>', '<b>Home<b>', '<b>Away<b>', '<b>Total<b>']
    labels = ['Won', 'Draw', 'Lost', 'Goals For', 'Goals Against', 'Shots For', 'Shots Against', 'T-Shots For','T-Shots Against', 'Points']
    
    index = labels.index(graph)
    
    fig = go.Figure(data=[go.Table(
        
        columnorder = [1,2,3,4],
        columnwidth = [60,40,40,40],
        
        header=dict(values=head,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[labels, home, away, total],
                   fill_color='lavender',
                   align='left'))
    ])   

    fig.update_layout(height=230, width=630, margin=dict(l=0, r=0, b=0,t=0))
    
    figx = go.Figure()
    
    figx.add_trace(
        go.Bar(
            x=['Home', 'Away', 'Total'],
            y=[home[index], away[index], total[index]],
            name=graph,
            text=[home[index], away[index], total[index]],
            textposition='auto'
        )
    )
    
    
    if (num==1):
         figx.update_traces(marker_color='rgb(255,185,15)', marker_line_color='rgb(205,149,12)', marker_line_width=1.5, opacity=0.6, texttemplate='%{text:.0f}')
    elif(num==2):
        figx.update_traces(marker_color='rgb(238,0,0)', marker_line_color='rgb(139,0,0)', marker_line_width=1.5, opacity=0.6, texttemplate='%{text:.0f}')
    else:
        figx.update_traces(marker_color='RGB(255,131,250)', marker_line_color='RGB(205,105,201)', marker_line_width=1.5, opacity=0.6, texttemplate='%{text:.0f}')
    
    figx.update_layout(
        xaxis=dict(autorange=True, title_text='Ground', title_font={"size": 10}, tickfont={"size":10}),
        yaxis=dict(autorange=True, title_text=graph, title_font={"size": 10}, tickfont={"size":10}),
        height=230,
        width=630,
        margin=dict(l=0, r=0, b=0,t=0),
        plot_bgcolor='rgb(255,255,255)',
    )
    
    graphs = [fig, figx]
    
    return graphs



#Function to display FPL team data
@st.cache(allow_output_mutation=True)
def teamFpl(elements, team):

    
    df = elements.loc[(elements['team'] == team)]
    df = df.sort_values(by=['positionName', 'bps_per_90'], ascending=True)
    
    df = df[['web_name', 'position', 'statusFull', 'strength', 'str-cat']]
    
    return df


# Function to calculate the Probability
@st.cache(allow_output_mutation=True)
def Probability(rating1, rating2):
    return 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating1 - rating2) / 400))



# Function to get team lineups
@st.cache(allow_output_mutation=True)
def lineup(elements, team):

    df = elements.loc[(elements['team'] == team) & (elements['statusFull'] == 'avail')]
    df = df.sort_values(by=['minutes', 'strength'], ascending=False)
    squad = df['web_name'].values
    lineup = squad[0:11]
    
    return lineup



# Function for points
@st.cache(allow_output_mutation=True)
def teamStrength(elements, team, line):
    
    df = elements.loc[(elements['team'] == team)]
    df = df.sort_values(by=['minutes', 'strength'], ascending=False)
    df = df[['web_name', 'strength']]
    filter1 = df.loc[(df['web_name'].isin(line))]
    score = filter1['strength'].sum()
    
    return score





@st.cache(allow_output_mutation=True)
def filterElements(elements, team):
    
    df = elements.loc[(elements['team'] == team)]
    df = df.sort_values(by=['positionName'], ascending=True)
    
    df = df[['web_name', 'position', 'now_cost', 'selected_by_percent', 'minutes', 'goals_scored', 'assists', 'saves', 'form', 
             'total_points', 'bps', 'ict_index']]
    return df




@st.cache(allow_output_mutation=True)
def filterPlayer(history, name):
    
    df = history.loc[(history['name'] == name)]
    
    df = df[['Date', 'opponent', 'minutes', 'goals_scored', 'assists', 'saves', 'total_points', 'bps', 'influence', 'creativity', 'threat', 'ict_index']]
    return df



@st.cache(allow_output_mutation=True)
def filterNames(elements, team):
    
    df = elements.loc[(elements['team'] == team)]
    df = df.sort_values(by=['positionName'], ascending=True)

    playerNames = df['web_name'].unique()
    
    return playerNames



@st.cache(allow_output_mutation=True)
def statshead2head(Teams, teama, teamb, games, graph):
    
    Teams = Teams[-games:]
    
    team1won = 0
    team1draw = 0
    team1lost = 0
    team1goalsfor = 0
    team1goalsagainst = 0
    
    team1shotsfor = 0
    team1shotsagainst = 0
    team1shotsTargetfor = 0
    team1shotsTargetagainst = 0
    
    team1points = 0
    
    team2won = 0
    team2draw = 0
    team2lost = 0
    team2goalsfor = 0
    team2goalsagainst = 0
    
    team2shotsfor = 0
    team2shotsagainst = 0
    team2shotsTargetfor = 0
    team2shotsTargetagainst = 0
    
    
    team2points = 0
    
    legth = Teams.shape[0]
    
    for i in range(0, legth):
        if (Teams['HomeTeam'].iloc[i] == teama):
            team1goalsfor += Teams['FTHG'].iloc[i]
            team1goalsagainst += Teams['FTAG'].iloc[i]
            
            team1shotsfor = Teams['HS'].iloc[i]
            team1shotsagainst = Teams['AS'].iloc[i]
            team1shotsTargetfor = Teams['HST'].iloc[i]
            team1shotsTargetagainst = Teams['AST'].iloc[i]
            
            if(Teams['FTR'].iloc[i] == 'H'):
                team1won += 1
                team1points += 3
            elif(Teams['FTR'].iloc[i] == 'D'):
                team1draw += 1
                team1points += 1
            else:
                team1lost += 1
                team1points += 0
                
        elif (Teams['AwayTeam'].iloc[i] == teama):
            team1goalsfor += Teams['FTAG'].iloc[i]
            team1goalsagainst += Teams['FTHG'].iloc[i]
            
            
            team1shotsfor = Teams['AS'].iloc[i]
            team1shotsagainst = Teams['HS'].iloc[i]
            team1shotsTargetfor = Teams['AST'].iloc[i]
            team1shotsTargetagainst = Teams['HST'].iloc[i]
            
            
            if(Teams['FTR'].iloc[i] == 'A'):
                team1won += 1
                team1points += 3
            elif(Teams['FTR'].iloc[i] == 'D'):
                team1draw += 1
                team1points += 1
            else:
                team1lost += 1
                team1points += 0
                
        if (Teams['HomeTeam'].iloc[i] == teamb):
            team2goalsfor += Teams['FTHG'].iloc[i]
            team2goalsagainst += Teams['FTAG'].iloc[i]
            
            team2shotsfor = Teams['HS'].iloc[i]
            team2shotsagainst = Teams['AS'].iloc[i]
            team2shotsTargetfor = Teams['HST'].iloc[i]
            team2shotsTargetagainst = Teams['AST'].iloc[i]
            
            if(Teams['FTR'].iloc[i] == 'H'):
                team2won += 1
                team2points += 3
            elif(Teams['FTR'].iloc[i] == 'D'):
                team2draw += 1
                team2points += 1
            else:
                team2lost += 1
                team2points += 0
                
                
        elif (Teams['AwayTeam'].iloc[i] == teamb):
            team2goalsfor += Teams['FTAG'].iloc[i]
            team2goalsagainst += Teams['FTHG'].iloc[i]
            
            team2shotsfor = Teams['AS'].iloc[i]
            team2shotsagainst = Teams['HS'].iloc[i]
            team2shotsTargetfor = Teams['AST'].iloc[i]
            team2shotsTargetagainst = Teams['HST'].iloc[i]
            
            if(Teams['FTR'].iloc[i] == 'A'):
                team2won += 1
                team2points += 3
            elif(Teams['FTR'].iloc[i] == 'D'):
                team2draw += 1
                team2points += 1
            else:
                team2lost += 1
                team2points += 0
                
                
    team1 = [team1won, team1draw, team1lost, team1goalsfor, team1goalsagainst, team1shotsfor,
             team1shotsagainst, team1shotsTargetfor, team1shotsTargetagainst, team1points]
    team2 = [team2won, team2draw, team2lost, team2goalsfor, team2goalsagainst, team2shotsfor,
             team2shotsagainst, team2shotsTargetfor, team2shotsTargetagainst, team2points]          
              
    head = ['<b>Statistic<b>', '<b>'+teama+'<b>', '<b>'+teamb+'<b>']
    labels = ['Won', 'Draw', 'Lost', 'Goals For', 'Goals Against', 'Shots For', 'Shots Against',
              'T-Shots For', 'T-Shots Against','Points']
    
    index = labels.index(graph)
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=head,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[labels, team1, team2],
                   fill_color='lavender',
                   align='left'))
    ])   

    fig.update_layout(height=230, width=630, margin=dict(l=0, r=0, b=0,t=0))
    
    
    figx = go.Figure()
    
    figx.add_trace(
        go.Bar(
            x=[teama, teamb],
            y=[team1[index], team2[index]],
            name=graph,
            text=[team1[index], team2[index]],
            textposition='auto'
        )
    )
    
    figx.update_traces(marker_color='RGB(255,106,106)', marker_line_color='RGB(205,85,85)', marker_line_width=1.5, opacity=0.6, texttemplate='%{text:.0f}')
    
    figx.update_layout(
        xaxis=dict(title_text='Team', title_font={"size": 10}, tickfont={"size":10}),
        yaxis=dict(title_text=graph, title_font={"size": 10}, tickfont={"size":10}),
        width=630,
        height=230,
        margin=dict(l=0, r=0, b=0,t=0),
        plot_bgcolor='rgb(255,255,255)',
    )
    
    graphs = [fig, figx]
    
    return graphs


######  Show table with target team statistics and Graph   ##########################################################################################
#####  with targetTeam:
    
st.markdown("<h3 style='text-align: left; color: purple; padding-left: 0px; font-size: 40px'><b>Team Analysis<b></h3>", unsafe_allow_html=True)

colx, coly, colz = st.columns([1, 1, 1])

tTeam = colx.selectbox('Team', teams, index=13)
lastGames = coly.number_input('Game History', min_value=2, step=1, value=6)
graph = colz.selectbox('Graphic', graph_options, index=5)
    
targTeam = tTeam
games=lastGames
graph_value = graph
    
homeTeam = merged_stats[(merged_stats['HomeTeam']==tTeam) | (merged_stats['AwayTeam']==tTeam)]

    
tarTeam = tables(homeTeam, games, tTeam)

colq, colw = st.columns([1, 1])

val = stats(homeTeam, targTeam, games, graph_value,0)
colq.plotly_chart(val[0], use_container_width=True)
colw.plotly_chart(val[1], use_container_width=True)

tarTeam

filter = filterElements(elements, tTeam)
playerStats = playerTables(filter)

playerNames = filterNames(elements, tTeam)

display = st.checkbox('Show Player Stats')

if display:
    st.markdown("<h4 style='text-align: left; color: #872657; padding-left: 0px; font-size: 30px'><b>Player Analysis<b></h4>", unsafe_allow_html=True)
    st.write(playerStats)
    name = st.selectbox('Detailed Player Statistics', playerNames, index=0)
    playerGames = filterPlayer(history, name)
    player = playerStatistics(playerGames, games)
    st.write(player)


######  Show table with team comparison statistics and Graphs with Head to Head Section   ############################################################
######with dataView:
       
st.markdown("<h3 style='text-align: left; color: purple; padding-left: 0px; font-size: 40px'><b>Team Comparison<b></h3>", unsafe_allow_html=True)

colx, coly = st.columns([1, 1])
    
home = colx.selectbox('Team 1', teams, index=13)
away= coly.selectbox('Team 2', teams, index=11)

teama = home
teamb = away

homeTeam12 = seasons[(seasons['HomeTeam']==home) | (seasons['AwayTeam']==home)]
awayTeam12 = seasons[(seasons['HomeTeam']==away) | (seasons['AwayTeam']==away)]   

homeTeam_merged = merged_stats[(merged_stats['HomeTeam']==home) | (merged_stats['AwayTeam']==home)]
awayTeam_merged = merged_stats[(merged_stats['HomeTeam']==away) | (merged_stats['AwayTeam']==away)]

    
head2head_merged = merged_stats[((merged_stats['HomeTeam']==home) & (merged_stats['AwayTeam']==away)) | ((merged_stats['HomeTeam']==away) & (merged_stats['AwayTeam']==home))]
    
target1 = team_merge[(team_merge['targetTeam']==home)]
team1 = pd.merge(homeTeam12, target1, how='inner', left_on=['gameDate','Season'], right_on=['gameDate','Season'])

target2 = team_merge[(team_merge['targetTeam']==away)]
team2 = pd.merge(awayTeam12 , target2, how='inner', left_on=['gameDate','Season'], right_on=['gameDate','Season'])

bothteams = pd.merge(team1 , team2, how='inner', left_on=['gameDate','Season'], right_on=['gameDate','Season'])
    
bothteams = bothteams[-games:]


cols, colt = st.columns([1,1])


val1 = stats(homeTeam_merged, teama, games, graph_value, 1)
val2 = stats(awayTeam_merged, teamb, games, graph_value, 2)

cols.plotly_chart(val1[0], use_container_width=True)
colt.plotly_chart(val2[0], use_container_width=True)


colu, colv = st.columns([1,1])
cols.plotly_chart(val1[1], use_container_width=True)
colt.plotly_chart(val2[1], use_container_width=True)

st.markdown("<h4 style='text-align: left; color: #872657; padding-left: 0px; font-size: 30px'><b>Head-to-Head<b></h4>", unsafe_allow_html=True)

test = tables(head2head_merged,games,teama) 

coli, colo = st.columns([1,1])
test2 = statshead2head(head2head_merged, teama, teamb, games, graph_value)
coli.plotly_chart(test2[0], use_container_width=True)
colo.plotly_chart(test2[1], use_container_width=True)

test
    
###################################################################################################################################################
    
homeFPL = teamFpl(elements, home)
awayFPL = teamFpl(elements, away)

home_names = homeFPL['web_name'].values
away_names = awayFPL['web_name'].values

homeline0 = lineup(elements, home)
awayline0 = lineup(elements, away)

Image.open('football_logo/football_logo.jpg')

soccerImage = Image.open('football_logo/football_logo1.png')
st.sidebar.image(soccerImage, width=200)
st.sidebar.markdown("<h3 style='text-align: left; color: #872657; padding-left: 0px; font-size: 30px'><b>Team Lineups<b></h3>", unsafe_allow_html=True)
st.sidebar.markdown(' ')

homeline = st.sidebar.multiselect(home+' Starting XI', home_names, default=homeline0)
homeCount = len(homeline)
homescore = round(teamStrength(elements, home, homeline),2)

st.sidebar.markdown("<h3 style='text-align: left; color: purple; padding-left: 0px; font-size: 20px'><b>Team Strength - "+str(homescore)+" | Players - "+str(homeCount)+"<b></h3>", unsafe_allow_html=True)


homeline = teamTables(homeFPL)

displayhome = st.sidebar.checkbox("Show "+home+" Players")

if displayhome:
    st.sidebar.plotly_chart(homeline)

awayline = st.sidebar.multiselect(away+' Starting XI', away_names, default=awayline0)
awayCount = len(awayline)
awayscore = round(teamStrength(elements, away, awayline),2)

st.sidebar.markdown("<h3 style='text-align: left; color: purple; padding-left: 0px; font-size: 20px'><b>Team Strength - "+str(awayscore)+" | Players - "+str(awayCount)+"<b></h3>", unsafe_allow_html=True)

awayline = teamTables(awayFPL)

displayaway = st.sidebar.checkbox("Show "+away+" Players")

if displayaway:
    st.sidebar.plotly_chart(awayline)
  
###############################################################################################################################################
premier, match = st.columns([1,4.5])

image = Image.open('football_logo/football_logo.jpg')
premier.image(image, use_column_width=True)

st.markdown(" ")
st.markdown(" ")

match.markdown("<h4 style='text-align: left; color: purple; padding-left: 0px; font-size: 40px'><b>Simulate Match<b></h4>", unsafe_allow_html=True)

homelogo = 'football_logo/'+home+'.png'
awaylogo = 'football_logo/'+away+'.png'

homeImage = Image.open(homelogo)
awayImage = Image.open(awaylogo)


colq, colp = st.columns([1,1])

colq.image(homeImage, width=200)
colq.markdown("<h3 style='text-align: left; color: #008080; padding-left: 0px; font-size: 20px'><b>Team Strength - "+str(homescore)+"<b></h3>", unsafe_allow_html=True)

colp.image(awayImage, width=200)
colp.markdown("<h3 style='text-align: left; color: #008080; padding-left: 0px; font-size: 20px'><b>Team Strength - "+str(awayscore)+"<b></h3>", unsafe_allow_html=True)

xgHome0= colq.number_input(home+' (Home Adjustment)', min_value=-5.0, max_value=5.0, step=0.1, value=0.0)
xgAway0= colp.number_input(away+' (Away Adjustment)', min_value=-5.0, max_value=5.0, step=0.1, value=0.0)

xgHome = homescore + xgHome0
xgAway = awayscore + xgAway0

proby = matchProb(xgHome, xgAway)


proHome = '{:.2%}'.format(round(proby[0],4))
proDraw = '{:.2%}'.format(round(proby[1],4))
proAway = '{:.2%}'.format(round(proby[2],4))

oddHome = round(1/proby[0],2)
oddDraw = round(1/proby[1],2)
oddAway = round(1/proby[2],2)  


st.markdown("<h3 style='text-align: left; color: #551A8B; padding-left: 0px; font-size: 30px'><b>Match Probability (Fair Odds)<b></h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1,1])

col1.markdown("<h3 style='text-align: left; color: #872657; padding-left: 0px; font-size: 20px'><b>HomeWin - "+str(proHome)+" ("+str(oddHome)+")<b></h3>", unsafe_allow_html=True)
col2.markdown("<h3 style='text-align: left; color: #872657; padding-left: 0px; font-size: 20px'><b>Draw - "+str(proDraw)+" ("+str(oddDraw)+")<b></h3>", unsafe_allow_html=True)
col3.markdown("<h3 style='text-align: left; color: #872657; padding-left: 0px; font-size: 20px'><b>AwayWin - "+str(proAway)+" ("+str(oddAway)+")<b></h3>", unsafe_allow_html=True)
    
###############################################################################################################################################
