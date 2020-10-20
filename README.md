# COVID_Forecast
A time series analysis and forecast for COVID-19 in the United States

# Objective

The primary objectives of this project are as follows:

* Use Time Series Analysis to forecast the number of COVID-19 positive patients in the most affected states by the end of October.  
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

The New York Times states that a majority of seasonal farmworkers are undocumented immigrants (*Farmworkers, Mostly Undocumented, Become ‘Essential’ During Pandemic*, Miriam Jordan, The New York Times).  The organization Student Action with Farmworkers states that 75% of farm workers were born in Mexico. 

What's interesting is that not only has the country of Mexico also been seeing the same summer trend we're investigating.

![mexico_summer](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/cases_mexico_covid_mx.PNG)

(Image Source: Instituciones del Gobierno de México) 

Earlier we discovered that farms begin increasing in early spring and peak during the summer, let's see what border crossings have been like during that time period.  

![cbp](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/border_crossings_cbp_gov.PNG)

(Image Source: United States Customs and Border Protection)

Although the numbers have been low, if we follow the red line (representing 2020), we find that there has been a steady increase in border apprehensions ever since April.  This does not imply, nor should we infer, that those people crossing the border were all carrying the virus.  We also don't have enough data to know where all of these workers ended up. We cannot infer that all of them began working on farms.  However, if many did end up working on farms, the major threat that it poses is that farms will become more crowded.  Which brings us to our next problem:

## Living Conditions of Undocumented Workers

The United States Department of Labor issues out H-2A visas to a number of immigrant workers every year to help work on farms.  These H-2A workers are considered guests and are entitled to adequate shelter and safe working environments provided by their employers.  Undocumented workers don't enjoy these same protections, unfortunately.  Many live in makeshift, or converted housing, such as this garage.

![living conditions](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/farmer_conditions_nyt.PNG)

Cramped quarters like these make it impossible to self-quarantine or maintain social distance.  This is terrible considering that these people are already at risk for respiratory infections due to their working conditions.

# Arizona

It would be hard to miss the increase in cases in northern Arizona.  Those counties don't have major farms, but there aren't major cities either.  So what is out there?

![reservations](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/reservations.PNG)

(Image Source: Google Maps)

Native American Reservations.  It's no secret that Native American Reservations aren't exactly the picture of prosperity.  The living conditions and health care available on Native American Reservations are quite poor.  According to Native Partnership, unemployment on Native American Reservations is 7 times higher than the US.  Homelessness and overcrowding are also major issues due to lack of resources to make proper housing. In the Navajo Nation, 15% of homes lack water, and about 33% of homes do not have adequate plumbing, kitchen facilities, and bedrooms.  All of these factors add up to Native Americans being at a much higher risk of catching and spreading coronavirus.

![infection_cdc.PNG](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/infection_cdc.PNG)

(Image Source: CDC.gov)

According to the CDC, Native Americans have the most hospitalizations for coronavirus out of all of the other demographics.  So the rapid spread of coronavirus and increase in cases in Arizona can most likely be attributed to the terrible living conditions of Native Americans. 

## The Root of the Problem

This infographic from the CDC points out the underlying problem to the spread of COVID-19: Socioeconomic Inequality.  People of color are often at much higher risk of contracting illness due to health care being too expensive.  Fear of not making enough money to pay the bills keeps them working when they should be home at resting when they're sick.  High rent prices lead to multiple families sharing the same home, again, making quarantine and social distancing impossible.

## Solutions

* Improved shelter and working conditions for ALL farm workers regardless of citizenship status
* Working together with Tribal Authorities to bring jobs to Native American Reservations. Particularly jobs that will help with housing, plumbing, and other amenities.  
* Improving the socioeconomic conditions for people of color needs to be made a top priority.

# Models

Now that we've addressed the exogenious factors contributing to the virus spread, let's address our primary objective and find out which modeling method works better: SARIMA or LSTM Neural Networks.

## Similarities 

### ACF and PACF Plots

![acf_pacf](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/fl_acf_pacf.PNG)

All of the states had similar higher orders with around AR(5) and MA(20)

### Decomposition

![decomp](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/fl_time_decomp.PNG)

The states all shared the same time decomposition with the only difference being New York's trend being in spring instead of summer.

Seasonality was roughly on a weekly basis for each state and the residuals were normally distributed until they reach their exponential trend.  During this period, they're residuals become heteroskedatic.

## Models

While models were made for all 5 of the states, for the sake of brevity, we'll be comparing the results of the states with our best performing SARIMA model (New York) to the state with the best performing LSTM Model (Georgia)

**Note**: Due to the different test sizes required for each method, use the following to compare the results of the test predictions of each model:

* The LSTM test sample size represent the dates September 21, 2020 - October 1, 2020.  These correspond with time skips 38 - 48 of the SARIMA Model

## Georgia

### SARIMA Test

![ga](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/ga_test.PNG)

As you can see, the SARIMA model isn't too bad.  It might not be on the mark, but it gets the general idea of the future trend.  What's the Root Mean Square Error?

![ga_rmse](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/ga_ts_test.PNG)

So our RMSE is accurate within 360 people, give or take.  

### SARIMA Forecast

![ga_forecast](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/ga_forecast.PNG)

The mean predictions show a slow decrease in cases, but let's check this against real data now that time has passed.

These figures are from the New York Times:

* Oct 5, 2020 - 741 Reported Cases

* Oct 12, 2020 - 906 Reported Cases 

* Oct 18, 2020 - 1,305 Reported Cases

So the mean predictions are off by a little each time.  An exogenous factor for why these number are increasing here (at least in my town) are the lines for early voting.  From what I've seen personally as I go for my morning runs are long voting lines with out much social distancing.  So this could be a factor that the model can't predict (again, at least in my area).

### LSTM

![ga_lstm](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/ga_fit.PNG)

![RMSE](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/ga_rmse.PNG)

Although the RMSE is only marginally better, you can see that the fit of the predictions is much better. It predicts the shape of the contour of the predictions is almost a perfect fit to the actual values.  

This is something that Georgia, Florida, and California have in common: The RMSE of the LSTM models were always better and the predictions were a better fit.

New York and Texas deviate from this pattern in that their LSTM models had larger errors than their corresponding SARIMA models, but the fit of the predictions were still much better with the LSTM models.

## New York

### SARIMA

![ny_test](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/ny_test.PNG)

![ny_rmse](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/ga_ts_test.PNG)

New York's has some incredible accuracy.  I believe this is due to the fact that since the other states had their exponential increase during the summer, the trend would get cut off during the Train-Test Split. Thus, the model would never see the entire trend, only the increasing portion.  New York, however, had the entire curve in the training set, which allows it to train better.

### SARIMA Forecast 

![ny_forecast](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/ny_forecast.PNG)

The forecast is very steady throughout the month of October.  What's odd is that this is the only model in which the mean predictions show a very subtle climb in the number of cases. 

Let's compare this to the New York Times reports:
* Oct 5, 2020 - 937 Reported Cases
* Oct 12, 2020 - 1,032 Reported Cases 
* Oct 18, 2020 - 1,390 Reported Cases

So in this case, the actual numbers are relatively close to the mean predictions.  Whether this trend continues is dependent on the residents and their governor. 

### LSTM

![ny_fit](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/ny_fit.PNG)

![nyrmse](https://raw.githubusercontent.com/bmauss/COVID_Forecast/master/images/ny_rmse.PNG)

Ok, so it's pretty obvious that New York actually had both the best SARIMA model and LSTM model, as well.  Georgia's was the best LSTM which out performed it's SARIMA model, and New York had a SARIMA model that out performed the LSTM in terms of errors.  The only other state to do this was Texas. 

Again, both New York's and Texas' SARIMA models had lower errors, but the LSTM predictions fit the trends much better.

# Conclusion

The reason LSTM Neural Networks have become so popular over SARIMA models is most likely because their predictions, even when wrong, will (generally) show the correct contour on the correct dates.  If this is truly the case, then this allows companies and governments to plan ahead of time and know when values will increase and when it will decrease.  No spikes will appear where drops occur.  Overall, LSTMs are a unique, simple, and elegant way to make time series predictions with.
