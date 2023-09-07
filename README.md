# promo-optimization_team-simpsons-paradox

Data Dictionary

Datasets:

There are 5 datasets that will be available to the participants:
-	sales_data.csv (primary dataset)
-	macro_data.csv
-	brand_segment_mapping_hackathon.csv
-	maximum_discount_constraint_hackathon.xlsx
-	volume_variation_constraint_hackathon.csv

sales_data.csv:

Contains the historical sales in HL per SKU, brand, pack type and pack size:

Columns:
-	Year: Historical year of the sale
-	Month: Historical month of the sale 
-	SKU: Unique ID of the SKU
-	Brand: Brand of the sale
-	Pack: Pack type of the SKU
-	Size: Pack Size of the SKU
-	Volume_Htls: Volume sold of the SKU in hectolitres
-	GTO_LC: Gross Turnover of the SKU in local currency
-	Promotional_Discount_LC: Discount given for promotion / promotion budget for the SKU in local currency
-	Other_Discounts_LC: Other discounts / budget for the SKU in local currency
-	Total_Discounts_LC: Sum of Promotional and Other discounts
-	Excise_LC: Excise taxes levied on the sale of the SKUs
-	Net_Revenue_LC: Revenue earned after deduction of costs (Net Revenue = GTO – Promotional discount – Other discount - Excise)
-	MACO_LC: Marginal Contribution of the SKU (MACO = Net Revenue - VILC) 
-	VILC_LC: Variable Inventory and Logistics cost of the SKU

macro_data.csv:

Contains the macroeconomic indicators for the market. Data is available at monthly level (certain macroeconomic indicators are refreshed on annual or quarterly).

Columns:
-	Year: Historical year of the macroeconomic KPI
-	Month: Historical month of the macroeconomic KPI
-	Retail_Sales_Index: A quarterly measure of the value of goods sold by retailers
-	Unemployment_Rate: Percentage of people in the labour force who are unemployed
-	CPI: Consumer Price Index (CPI) is a measure of the average change overtime in the prices paid by urban consumers for a market basket of consumer goods and services
-	Private_Consumption: It is a measure of all the money spent by consumers in the country to buy goods and services
-	Gross_Domestic_Saving: It is a measure of savings of household sector, private corporate sector, and public sector.
-	Brad_Money: It measures the amount of money circulating in an economy
-	GDP: Gross Domestic Product is a monetary measure of the market value of all the final goods and services produced in a specific time-period by a country or countries.

brand_segment_mapping_hackathon.csv:

Contains the mapping between brand and price segment

Columns:
-	Brand: Name of the brand
-	PriceSegment: Name of the price segment that the corresponding brand belongs to

maximum_discount_constraint_hackathon.csv:

Contains the maximum discount that can be given when finding the optimal budget. There are three sheets:

Brand – Contains the maximum discount that can be given per brand:

Columns:
-	Brand: Name of the brand
-	max_discount: Maximum discount for the corresponding brand

Pack – Contains the maximum discount that can be given per pack type:

Columns:
-	Pack: Pack type
-	max_discount: Maximum discount for the corresponding pack

PriceSegment – Contains the maximum discount that can be given per price segment:

Columns:
-	PriceSegment: Price segment
-	max_discount: Maximum discount for the corresponding price segment



volume_variation_constraint_hackathon.csv

Contains the minimum and maximum variation in volume allowed per month at SKU, Brand, Pack and Size level, when finding the optimal budget

Columns:
-	SKU: Unique ID of the SKU
-	Brand: Name of the brand
-	Pack: Pack type of the SKU
-	Size: Pack size of the SKU
-	Minimum Volume Variation: Minimum allowed change in volume compared to previous month for the optimal budget
-	Maximum Volume Variation: Maximum allowed change in volume compared to 
