## Mendelian Randomization 

# - **Project:** DementiaSeq (AKA: Resolution) Project
# 	- **Author(s):** Sara Bandres-Ciga
# 	- **Date Last Updated:** 9.05.2019

### Methods description:

# Mendelian randomization (MR) is an epidemiological method that can be used to provide support for causality between a modifiable exposure/phenotypic trait ( termed exposure) and a disease outcome. Two-sample MR was used to provide support for causality between 401 exposures and the risk of both DLB and FTD. MR analyses were performed using the MR Base package version 3.2.2; [https://github.com/MRCIEU/TwoSampleMR])(Hemani et al., 2017).
# We used the following stringent criteria for any exposure GWAS to be included in our analysis: (i) the GWAS had to report SNPs with p-values less than 5.0x10-8 for their association with a given exposure; (ii) these SNPs or their proxies (linkage disequilibrium R2 value >= 0.8) had to be present in both the exposure and outcome (PD) datasets; (iii) these SNPs were independent signals that were generated through a process called ‘clumping’. In order to ‘clump’, index SNPs were identified by ranking exposure associations from the smallest to largest p-value (but still with a cutoff value of p=5x10−8). Clumped SNPs were those in linkage disequilibrium (LD) with index SNPs (R2 threshold of 0.001) or within 10,000 kb physical distance.  Hence, each index SNP represented a number of clumped SNPs that were all associated with or near to the index SNP, and the index SNPs were all independent of one another (according to the stringent parameters defined here). Then, we further expanded our filtering approach as follows: (iv) in order to use MR sensitivity analyses designed to identify pleiotropy, each GWAS had to include a minimum of ten associated SNPs; (v) the number of cases was > 250 for GWASes of a binary exposure or > 250 individuals for GWASes of a continuous exposure; and (vi) both the exposure and the outcome data were drawn from European populations. A total of 401 traits met our filtering criteria, consisting of 175 published GWASes and 226 unpublished GWASes from UK Biobank (UKB; [www.ukbiobank.ac.uk](http://www.ukbiobank.ac.uk/)).
# Harmonization was undertaken to rule out strand mismatches and to ensure alignment of SNP effect sizes. Within each exposure GWAS, Wald ratios were calculated for each extracted SNP by dividing the per-allele log-odds ratio (or beta) of that variant in the PD GWAS data by the log-odds ratio (or beta) of the same variant in the exposure data.
# First, the inverse-variance weighted (IVW) method was implemented to examine the relationship between the individual exposures and PD. Effects of pleiotropy for each analysis were studied by first looking for evidence of heterogeneity in the individual SNP Wald ratios and then undertaking a range of sensitivity analyses, each with different underlying assumptions. Heterogeneity in the IVW estimates was tested using the Cochran’s Q test, quantified using the I2 statistic, and displayed in forest plots. IVW radial analysis was performed as a complementary method to account for SNPs acting as heterogeneous outliers and to determine the effect of resulting bias on the IVW estimate
# For effect estimate directionality, odds ratios were scaled on a standard deviation increase in genetic risk for the exposure from that population mean. We evaluated the possibility that the overall estimate was driven by a single SNP using leave-one-out (LOO) analyses for each of the phenotypic traits associated with PD. Finally, we tested for reverse causation by using SNPs tagging the independent loci described in the latest PD GWAS as exposure instrumental variables, and exposure GWASes as the outcomes.

###########################################################################################################################################

library(data.table)
library(TwoSampleMR)
install.packages("MRinstruments")
library(MRInstruments)
install.packages("WSpiller/RadialMR")
library("RadialMR")

### STAGE ONE: Format SumStats

#1): Create data.table/apetree objects 'DLBtemp' & 'HRCtemp' by passing correct directory path to read in file. 
DLBtemp <- fread("/data/LNG/RESOLUTION/MR/DLB.tbl", header = T)
HRCtemp <- fread("/data/LNG/RESOLUTION/MR/chrBpIndexRs.tab", header = F)


#2): Set the list of names/labels in HRCtemp returned by the name() function to a new string vector. 
names(HRCtemp) <- c("index","SNP","b")


#3): Create subset 'HRC' of data.frame 'HRCtemp' by excluding any vector with SNP value of "." This function selects any element that doesn't contain that value (and creates a new data.frame object) 
HRC <- subset(HRCtemp, SNP != ".")


#DLBtemp$index <- paste(DLBtemp$chr,DLBtemp$bp,sep = "_")


#4): Replace the values in the 'index' column of DLBtemp with the values in the 'MarkerName' column, and then create new data.frame 'DBL' by combining all the rows in 'DLBtemp' & 'HRC' that have the same value in the 'index' column.
DLBtemp$index <- DLBtemp$MarkerName
DLB <- merge(DLBtemp, HRC, by = "index")


#5): Make additional modifications to newly created DLB data.frame - coerse values in the 'Allele1' and 'Allele2' columns to the character type, make all characters upper-case, and assign those values to the 'effect_allele' and 'other_allele' columns within DLB, respectively.
DLB$effect_allele <- toupper(as.character(DLB$Allele1))
DLB$other_allele <- toupper(as.character(DLB$Allele2))


#6): We need to format the phylogenetic tree represented by the infile. The infile may be in 'ape' or 'ouch' format, and the function format_data() will format it as an 'ouch' tree object. Both tree formats comes from packages that specialize in phylogenetic analysis and can be represented as matrices (data.frame objects). However, the 'ouch' format allows for use of methods of Class ouchtree to access elements of tree objects instead of accessing them directly (via the $ operator, as seen above), and allows for calculation of a variance-covariance matrix. In this way, 'Out_data' becomes an ouchtree object, taking our modified 'DLB' apetree/data.frame and adding additional columns for values of impending MR analysis. Note that in both of these phylogenetic formats, trees can be represented as matrices - I.E., data.frames!
Out_data <- format_data(DLB, type="outcome", beta_col = "Effect", se_col = "StdErr", eaf_col = "Freq1", pval_col = "P-value")

### STAGE TWO: Run MR analysis

#token <- get_mrbase_access_token()

#1): In the future, we may use the above function to authenticate our access to the MR database (to search for GWAS-significant SNPs), but for now we hard-code an access token (presumably b/c folks at NIH have access thru Biowolf). We use the available_outcomes() method on our token to get details about all the available GWAS studies on the MR database, saving that to data.table file 'possibleInstruments.'
token <- "ya29.Glt2BSKxkLWJCzEV6_L_QL9fEfQnMdHDnAGeOy-4wl-6sSFz2HuyV2azTIlI73Wa9Qiehd2b7UhjyFDi1BcP-Uwd2xj3OnJAQKp-xZnOHvySYSjisnLdhOOVYaye"
possibleInstruments <- available_outcomes(access_token = token)

#2): Now we begin cross-examination of GWAS studies and your personal data. In this instance, we create a data.table object the includes information on MR traits from the LNG lab's GWAS data - use your own data here. Note that we've hard coded the for-loop, but the number of GWAS IDs in your own studies may vary.
listOfGwasIds <- read.table("/data/LNG/RESOLUTION/MR/TRAITS.txt", header = T)
for(i in 1:401)
{
    
#3): For every element (and each iteration of this loop) in the GWAS ID column from your dataset, we assign the ID in character type to instrumentID, and create a string type variable 'tag' to mark it's location within your dataset, printing it to the console.
  instrumentId <- as.character(listOfGwasIds$id[i])
  tag <- paste("INSTRUMENT IS ",instrumentId," AT i = ",i, sep = "")
  print(tag)
  flagged <- "nope"
    
#4): In order to compare your data with pulled GWAS data, start by creating a data.table that stores data relevant to your own data. The key here is the 'outcomes' argument, which we previously set to include all the GWAS IDs in your dataset. The other parameters may be adjusted to further refine the data extraction. p1 = P-value threshold for keeping a SNP. clump = Whether or not to return independent SNPs only (default=TRUE). r2 = The maximum LD R-square allowed between returned SNPs. kb = The distance in which to search for LD R-square values. All the parameters default in such a way as to not restict the search - note that assigning a parameter for p1 allows you to obtain SNP effects for constructing polygenic risk scores.
  Exp_data <- extract_instruments(outcomes=instrumentId, p1 = 5e-08, clump = TRUE, p2 = 5e-08,
                                  r2 = 0.001, kb = 10000, access_token = token,
                                  force_server = TRUE)

    
#5): We use logical assessment to determine whether there are empty values in the beta.exposure column from the GWAS extract 'Exp_data' data.table - in those cases, we'll skip comparison between that row of data and our own GWAS data.
  skip <- ifelse(length(Exp_data$beta.exposure) < 1, 1, 0)
  if(skip == 0)
  {
      

#6): We create a new data.frame 'dat' that combines the exposure data and outcome data in pairwise fashion; each exposure is matched with every other outcome in a pair, and vice versa. Then, data.frame 'res' will contain data for every MR comparison procedure between those pairs. We specify two other data.frames 'het' and 'ple' to represent whether the pairs are highly heterogenous, and whether directional horizontal pleiotropy is driving the results of our MR analysis. From here, consider saving results of your MR analysis into three separate txt files in your project folder for reference.
   dat <- harmonise_data(exposure_dat=Exp_data, outcome_dat=Out_data, action=2)
   res <- mr(dat)
   print(res)
   het <- mr_heterogeneity(dat)
   print(het)
   ple <- mr_pleiotropy_test(dat)
   print(ple)
   write.table(res, file = paste("/data/LNG/RESOLUTION/MR/",instrumentId,"_1res.txt",sep = ""), quote = F, sep = ",")
   write.table(het, file = paste("/data/LNG/RESOLUTION/MR/",instrumentId,"_2het.txt",sep = ""), quote = F, sep = ",")
   write.table(ple, file = paste("/data/LNG/RESOLUTION/MR/",instrumentId,"_3ple.txt",sep = ""), quote = F, sep = ",")
      

#7): 'res_single' will contain results similar to 'res', however the MR analyses for each exposure-outcome pair is conducted multiple times for all available SNPs. You can adjust arguments in the mr_singlesnp() function to conduct Wald's Test, IVW and MR Egger methods, or fixed-effect meta-analysis. Save the results into another txt.file.
   res_single <- mr_singlesnp(res)
   write.table(res_single, file = paste("/data/LNG/RESOLUTION/MR/",instrumentId,"_4res_single.txt",sep = ""), quote = F, sep = ",")
      
      
#8): We use a "leave-one-out" analysis (conduct MR accounting for all but one SNP [a total number of times equal to the number of available SNPs such that each one is left out once] to determine if a single SNP is driving the results of the MR analysis. The directionality test provides the directional metric of the relationship between each exposure-outcome pair. It is reccomended to view results of "leave-one-out" analysis graphically, but saving the numerical results to a new file is good practice.
   out <- directionality_test(dat)
   res_loo <- mr_leaveoneout(dat)
   p3 <- mr_leaveoneout_plot(res_loo)
   p3[[1]]
   fwrite(res_loo, file = "/data/LNG/RESOLUTION/MR/", na = "NA", quote = F, row.names = F, sep = "\t")

      
#9): The format_radial method essentially reads in summary data from MR analysis. Arguments include vectors of "beta-coefficient" values and standard errors of those values. We pull a vector of names of the used variants from SNP, and assign all these values to a new data.frame 'r_input,' then used by a series of other methods to eventually plot a "funneled diagram" of the vector of the ivw and egger MR estimates.
   r_input <- format_radial(BXG = dat$beta.exposure, BYG = dat$beta.outcome, seBXG = dat$se.exposure,seBYG =dat$se.outcome, RSID = dat$SNP)
   ivw <- ivw_radial(r_input,0.05,2)
   egger <- egger_radial(r_input,0.05,2)
   plot_radial(c(ivw,egger),TRUE,TRUE,FALSE)
   funnel_radial(c(ivw,egger),TRUE)

      
#10): Conduct MR forest_plot analysis to finish up - after this loop runs for all i, skip will be set to FALSE and the loop will end.
   p2 <- mr_forest_plot(res_single)
   p2
  }
  else
  {
    print("FAIL")
  }
}
#q(no)

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU1NDUyMjUzNV19
-->