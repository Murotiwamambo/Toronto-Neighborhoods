#!/usr/bin/env python
# coding: utf-8

# #  Segmenting and Clustering Neighborhoods in Toronto
# 
# #### In this exercise, i will cluster all the neighbhorhoods in Toronto to find those that are similar. I chose to use all the neighborhoods to have a complete picture of the city. 

# ## 1. Scrapping the web for Toronto Neighborhoods

# In[1]:


#importing packages

import urllib.request # import the library we use to open URLs
from bs4 import BeautifulSoup # import the BeautifulSoup library so we can parse HTML and XML documents
import pandas as pd
import numpy as np


# In[2]:


# specify which URL/web page we are going to be scraping
url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'

# open the url using urllib.request and put the HTML into the page variable
page = page = urllib.request.urlopen(url)


# In[3]:


# parse the HTML from our URL into the BeautifulSoup parse tree format
soup = BeautifulSoup(page, "lxml")

#unimportant commands but were used to verify the name of website i was scrapping
#print(soup.prettify())
#capturing the title of the page 
#soup.title 
#capturing the title of the page without the xml tags
#soup.title.string

# use the 'find_all' function to bring back all instances of the 'table' tag in the HTML and store in 'all_tables' variable
all_tables=soup.find_all("table")

# extracting only the right table using its class 
right_table=soup.find('table', class_='wikitable sortable') 

#a loop through the rows to get values for each column
Postal_Code=[]
Borough=[]
Neighbourhood=[]

for row in right_table.findAll('tr'):
    cells=row.findAll('td')
    if len(cells)==3:
        Postal_Code.append(cells[0].find(text=True))
        Borough.append(cells[1].find(text=True))
        Neighbourhood.append(cells[2].find(text=True))
       

df = pd.DataFrame(Postal_Code, columns = ['Postal_Code'])
df['Borough'] = Borough
df['Neighbourhood'] = Neighbourhood

#cleaning up the dataframe
df = df.replace('\\n', '',regex=True)
df.replace('',np.nan,inplace = True )
df.dropna(subset=['Neighbourhood'], inplace = True)
df.head()


# In[4]:


df.shape


# ## 2. Merging the data with location coordinates

# In[5]:


#packages 

#!conda install -c conda-forge geocoder --yes # uncomment this line if you haven't completed the Foursquare API lab
#from geopy.geocoders import Nominatim 

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library
import requests # library to handle requests


# In[6]:


#importing the csv file with lattitude and longitude
lat_long_df = pd.read_csv('Geospatial_Coordinates.csv')
lat_long_df.rename(columns = {'Postal Code':'Postal_Code'}, inplace = True) 

#merging the data
df = pd.merge(df,lat_long_df, on='Postal_Code')
df.head(10)


# ## 3. Mapping the data and the neighborhoods

# In[7]:


# create map of Toronto using latitude and longitude values
latitude = 43.6532
longitude = -79.3832
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df['Latitude'], df['Longitude'], df['Borough'], df['Neighbourhood']):
    label = '{}, {}'.format(df, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# ## 4. Exploring the neighborhoods

# In[8]:


#Define Foursquare Credentials and Version


# In[9]:


CLIENT_ID = 'VPJH3UW2X3G0FFMWBML5NHEKR0J0SZULBUBTPUHBQCEW4ABQ' # your Foursquare ID
CLIENT_SECRET = 'F2LOGCGRPP0D3PKI0UYKSZ2A02NYNXDGTJC5T1VYVXRZ1E4J' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[10]:


#2. Explore Neighborhoods in Manhattan
#Let's create a function to repeat the same process to all the neighborhoods in Manhattan


# In[11]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[12]:


toronto_venues = getNearbyVenues(names=df['Neighbourhood'],
                                   latitudes=df['Latitude'],
                                   longitudes=df['Longitude']
                                  )


# In[13]:


#print(toronto_venues.shape)
#toronto_venues.head(10)

#Let's check how many venues were returned for each neighborhood
toronto_venues.groupby('Neighbourhood').count().head(10)

#Let's find out how many unique categories can be curated from all the returned venues
print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# #### Analyzing Each Neighborhood

# In[14]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighbourhood'] = toronto_venues['Neighbourhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

#toronto_onehot.head()
toronto_onehot.shape

#Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category¶
toronto_grouped = toronto_onehot.groupby('Neighbourhood').mean().reset_index()
toronto_grouped

#Let's confirm the new size
toronto_grouped.shape


# In[15]:


num_top_venues = 5

for hood in toronto_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[16]:


#First, a function to sort the venues in descending order.
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[17]:


# creatinf the new dataframe and display the top 10 venues for each neighborhood.

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighbourhood'] = toronto_grouped['Neighbourhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()



# ## 5. Clustering  Neighborhoods 
# 
# #### Running the k-means algorithm to cluster the neighborhood into 6 clusters.

# In[18]:


# set number of clusters
kclusters = 6

toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 

# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = df

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

#Finally, let's visualize the resulting clusters
toronto_merged.dropna(subset=['Cluster Labels'], inplace = True)
toronto_merged.head() 


# #### Visualizing the clusters of the neighborhoods

# In[19]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
colors_array = colors_array.astype(int)
#rainbow = [colors.rgb2hex(i) for i in colors_array]

rainbow = ['#0000ff', '#00ff00', '#000000', '#ff0000', '#ffA500','#FFFF00' ]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster-1)],
        fill=True,
        fill_color=rainbow[int(cluster-1)],
        fill_opacity=0.7).add_to(map_clusters)

    
       
map_clusters


# In[ ]:





# In[20]:


# for more instructions on how to scrap web data
# https://simpleanalytical.com/how-to-web-scrape-wikipedia-python-urllib-beautiful-soup-pandas

