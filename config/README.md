## Data Dictionary

### Available Datasets

There are 5 datasets available to participants:
1. `sales_data.csv` (primary dataset)
2. `macro_data.csv`  
3. `brand_segment_mapping.csv`
4. `maximum_discount_constraint.xlsx`
5. `volume_variation_constraint.csv`

### Dataset Details

#### 1. Sales Data (`sales_data.csv`)
Contains historical sales in HL per SKU, brand, pack type and pack size.

**Columns:**
| Column | Description |
|--------|-------------|
| `year` | Historical year of the sale |
| `month` | Historical month of the sale |
| `sku` | Unique ID of the SKU |
| `brand` | Brand of the sale |
| `pack` | Pack type of the SKU |
| `size` | Pack Size of the SKU |
| `volume_hl` | Volume sold of the SKU in hectolitres |
| `gto` | Gross Turnover of the SKU in local currency |
| `promotional_discount` | Discount given for promotion / promotion budget for the SKU in local currency |
| `other_discount` | Other discounts / budget for the SKU in local currency |
| `total_discounts` | Sum of Promotional and Other discounts |
| `excise` | Excise taxes levied on the sale of the SKUs |
| `net_revenue` | Revenue earned after deduction of costs (Net Revenue = GTO – Promotional discount – Other discount - Excise) |
| `maco` | Marginal Contribution of the SKU (MACO = Net Revenue - VILC) |
| `vilc` | Variable Inventory and Logistics cost of the SKU |

#### 2. Macroeconomic Data (`macro_data.csv`)
Contains macroeconomic indicators for the market at monthly level (some indicators refreshed annually/quarterly).

**Columns:**
| Column | Description |
|--------|-------------|
| `year` | Historical year of the macroeconomic KPI |
| `month` | Historical month of the macroeconomic KPI |
| `retail_sales_index` | A quarterly measure of the value of goods sold by retailers |
| `unemployment_rate` | Percentage of people in the labour force who are unemployed |
| `cpi` | Consumer price index; measure of the average change overtime in the prices paid by urban consumers |
| `private_consumption` | Measure of all the money spent by consumers in the country to buy goods and services |
| `gross_domestic_saving` | Measure of savings of household sector, private corporate sector, and public sector |
| `brad_money` | Measures of amount of money circulating in an economy |
| `gdp` | Gross domestic product; monetary measure of market value of all final goods and services |

#### 3. Brand Segment Mapping (`brand_segment_mapping.csv`)
Maps brands to their price segments.

**Columns:**
| Column | Description |
|--------|-------------|
| `brand` | Brand name |
| `price_segment` | Corresponding price segment |

#### 4. Maximum Discount Constraints (`maximum_discount_constraint.xlsx`)
Contains maximum allowable discounts across three dimensions in separate sheets:

**Brand Sheet:**
| Column | Description |
|--------|-------------|
| `brand` | Brand name |
| `max_discount` | Maximum allowed discount for brand |

**Pack Sheet:**
| Column | Description |
|--------|-------------|
| `pack` | Pack type |
| `max_discount` | Maximum allowed discount for pack type |

**Price Segment Sheet:**
| Column | Description |
|--------|-------------|
| `price_segment` | Price segment |
| `max_discount` | Maximum allowed discount for segment |

Note: Synthetic data generator will create this as 3 different CSVs.

#### 5. Volume Variation Constraints (`volume_variation_constraint.csv`)
Defines volume variation constraints per SKU for optimal budget calculation.

**Columns:**
| Column | Description |
|--------|-------------|
| `sku` | Unique SKU ID |
| `brand` | Brand name |
| `min_volume_variation` | Minimum allowed monthly volume change |
| `max_volume_variation` | Maximum allowed monthly volume change |
