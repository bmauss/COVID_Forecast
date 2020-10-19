# COVID_Forecast
A time series analysis and forecast for COVID-19 in the United States

# Objective

The primary objectives of this project are as follows:

* Use Time Series Analysis to forecast the number of COVID-19 positive patients in the United States in the coming months.  

* Determine which models are better suited for making these predictions: ARIMA or Long Short-Term Mermory (LSTM) Recurrent Neural Networks (RNNs).

The secondary objective:

* Find any social circumstances that could attribute to increase in cases.

# Obtaining Data

COVID-19 is a highly contagious respiratory infection.  Symptoms vary by type and severity depending on immune system strength and age.  Despite this, most people generally report flu-like symptoms.  While contracting the illness can happen as remotely as contact with an object touched by a person with the sickness, the primary mode of transmission is from person-to-person contact, followed by airborne germs. 

While everyone is at risk of contracting the illness, it is primarily the elderly and people with compromised immune systems that are at risk of death.

# What States Have the Most Cases?

![Imgur](https://i.imgur.com/XuZwCzY.png)

So, our top 5 states that we'll look into are **California**, **Florida**, **Georgia**, **New York**, and **Texas**.

## How are they currently Trending?

![GitHub](https://github.com/bmauss/COVID_Forecast/blob/master/images/state_trend.PNG)

So New York has been on the downward slope and actually seems to have stabilized drastically since spring.  What is interesting is that our **other four states**  have been seeing an **increase in cases starting around mid-June**, peaking in **mid-July**, and **sloping off in August** as we go into fall.  

## What's something that happens during the Summer?

![hiring_season_epi_org](https://github.com/bmauss/COVID_Forecast/blob/master/images/hiring_season_epi_org.PNG)

(Image source: Economic Policy Institute)

Many crops are picked and harvested in the summer.  According to the United States Department of Agriculture, Texas and California are two of the largest food producers in the country.  Do our other 3 states have profitable farms?

![richest_farms](https://github.com/bmauss/COVID_Forecast/blob/master/images/richest_farms_usda.PNG)

(Image Source: USDA, Agricultural Census 2017)

From this graphic, we see that there are some pretty profitable farms in southern Georgia and Florida.  New York has some, too.  There's also a large cluster of farms along the Mississippi River, as well as throughout the Midwest.  Let's watch the spread of COVID-19 starting at the end of May and got through the end of August.  To account for population density, we'll divide the number of COVID cases in a county by its estimated population in 2019 (estimates provided by the US Census Bureau).  **Note**: Because these are estimates, we should interpret very dark areas as counties where a large number of its residents have been confirmed with COVID-19, not as an indication that 100% of its residents contracting the virus.

### May
![may](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/may_confirmed.PNG)

Notice how in the southwestern corner of Georgia and the northern-most border of Texas there are already counties reporting a large amount of their population as confirmed with coronavirus.  These areas also correlate with counties that have highly profitable farms.  Let's move on to June.

### June
![June](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/june_confirmed.PNG)

Ok, more of the farming communities in Texas are getting worse, and the virus is making its way up the Mississippi River.  A few more counties in the Midwest are lighting up and the farms along California's southern border is also showing an increase in cases.

### July
![July](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/july_confirmed.PNG)

It's definitely spreading out from the Mississippi River at this point.  Southern and central Florida has gotten worse, more farming communities in northern Texas and the Midwest are deteriorating, and the virus has hit the Fresno, California area.  Also, North Carolina, another top producer, has gotten much worse. Myrtle Beach, South Carolina isn't looking too good either.  

### August
![August](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/august_confirmed.PNG)

Ok, if we ignore the fact that the entire southeast coast looks like it's been overrun by fire ants, you can see that the Midwest farming counties are getting worse and the Fresno area deteriorated quickly.

There does seem to be a correlation between the spread of coronavirus over the summer and highly profitable farms.  This doesn't mean that the food supply is tainted, only that the workers are at risk.  In fact, according to American Progress, farmworkers are at higher risk of respiratory infections due to handling aerosol chemicals and pesticides to treat the crops.   

## Who do Farms Hire?

The New York Times states that 
