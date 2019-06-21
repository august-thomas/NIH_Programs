library(data.table)
library(dplyr)
library(tidyr)
library(plyr)

###########################################################################################################################################
# FORWARD ---
# THIS SCRIPT TAKES A .TXT FILE CONTAINING ROH VALUES DELINEATED BY COLONS AND CALCULATES PROPORTION TESTS ON PAIRWISE ROH INSTANCES. IT WRITES TWO TAB-DELINEATED .TXT FILES - "ROH_subsetted.txt" IS AN ARRAY OF INSTANCES FROM A SUBSET OF AN INITIAL DATAFRAME IN WHICH BOTH THE PROPORTION OF INFILE ROH CASES IS GREATER THAN THE NUMBER OF INFILE ROH CONTROLS, AND RESULTS OF THE PROPORTION TEST FOR THAT GIVEN INSTANCE IS SMALLER THAN THE QUOTIENT OF 0.05 AND TOTAL NUMBER OF PROPORTION TESTS RUN FROM THE INFILE. "ROH_caseEnriched.txt" IS AN ARRAY OF INSTANCES FROM A SUBSET OF AN INITIAL DATAFRAME WHERE THE PROPORTION OF CASES ARE GREATER THAN THE PROPORTION OF CONTROLS.
###########################################################################################################################################


# First set a working directory - program must be able to read in ROH txt file. Read in the .txt file in the working directory to create a data.table object 'ROH'.
setwd("~/Desktop/")
ROH <- fread("ROH_tocheck.txt", header = T)


# For each element of the data table, call the following method (strsplit) and turn the result into character list 'case'. The strsplit method will split factors of all character strings passed by first argument 'x' by the ":" character. In this case, 'x' --> as.character(ROH$case_to_control), which means that each element in a list 'case_to_control' is coerced into the character datatype and searched for the ":", where a split is made. Note that ldply()[[1]] / ldply()[[2]] determine the indices in the ROH datatable to conduct this process; 1 for the case data and 2 for the control. This is done for two reasons - 1): We can't call the as.numeric() method on factors - only on strings. 2): Messy data converted to character strings can be searched and delineated by the strsplit() method easily.
ROH$case <- ldply(strsplit(as.character(ROH$case_to_control), split = ":"))[[1]]
ROH$control <- ldply(strsplit(as.character(ROH$case_to_control), split = ":"))[[2]]


# Above, we forced every factor in data.table ROH to become lists of the character data type (as part of the split procedure) - now we revert those strings back into numeric data type - for both the Ncase and Ncontrol character lists.
ROH$NcasesWithROHs <- as.numeric(ROH$case)
ROH$NcontrolsWithROHs <- as.numeric(ROH$control)

# Here we create a copy of the numeric Ncase and Ncontrol lists above. The goal being to add them together into a numeric factor. I'm wondering why this is necessary though - there doesn't seem to be any interaction between the NcasesWithROHs list and ROH$caseN list for example. Why not create this factor from the original?
ROH$caseN <- as.numeric(ROH$case)
ROH$controlN <- as.numeric(ROH$control)
ROH$combinedN <- ROH$caseN + ROH$controlN


# Here we initialize an empty list 'P' - it will eventually include proportion test results.
ROH$P <- NA


# With our objects assigned, we're now ready to conduct exact/approximate tests for proportions from the infile. We'll first conduct a prop.test for every item in the 'case' list (which will also be the number of cases in the original infile).
for(i in 1:length(ROH$case)
    
    {  
# We vectorize the case and control portions of the original infile that include ROH proportions. Each proportion test relies on vectorization of the correct [hardcoded?] values for the number of cases and control instances. If the argument for sample size isn't passed correctly there will be an error. We'll assign the output of each proportion test to the empty column 'P' in the ROH data.frame.
  thisP <- prop.test(x = c(ROH$NcasesWithROHs[i], ROH$NcontrolsWithROHs[i]), n = c(1070.00, 4711.0))
  ROH$P[i] <- thisP$p.value  
}

# With the P column in the ROH data.frame constructed, we now assign values to column 'total_ROH_count' by simply making a factor out of previously initialized 'caseN' and 'controlN' numeric list objects - we now fill in values for two new columns by dividing the 'caseN' and 'controlN' portions of the factor by the number of cases/controls in the original infile to create lists containing the proportion that each value is out of the whole ('propCases' and 'propControls').
ROH$total_ROH_count <- ROH$caseN + ROH$controlN
ROH$propCases <- ROH$caseN/1070.00
ROH$propControls <- ROH$controlN/4711.0


# We use an ifelse() function to logically separate outcomes of numerical comparison of case versus control instances into two separate indexes in a column in ROH named 'caseEnriched.' In instances where the proportion of cases > controls, the instance is assigned to the 1st index; else, the instance is assigned to the 0th index (and is dropped from the data.frame).
ROH$caseEnriched <- ifelse(ROH$propCases > ROH$propControls, 1, 0)


# We return a subset of the original ROH data.frame object. Elements are added if they meet two conditions - 1st: the zeroth element can not be added. 2nd: the element must be in the 1st index of the caseEnriched list.
ROH_subsetted <- subset(ROH, total_ROH_count >= 1 & caseEnriched == 1)


# Create integer 'Ngenes,' set as the number of elements in the ROH_subsetted data.frame. Subset the data.frame object once again into 'passMultiTest' list by testing whether the result of that instance's proportion test is less than 0.05 divided by the number of elements in the parent data.frame object. If true, assign to 1st index of passMultiTest list - if false, assign to 0th index (dropping as before).
Ngenes <- length(ROH_subsetted$PD)
ROH_subsetted$passMultiTest <- ifelse(ROH_subsetted$P <= (0.05/Ngenes),1,0)


# From here, we write the accepted indicies of the fina ROH data.frame (that met all criterion), as well as all the instances where the propCase was larger than the propControl to two new .txt files, named accordly.
subset(ROH_subsetted, passMultiTest == 1)
write.table(ROH_subsetted, file = "ROH_subsetted.txt", quote = F, row.names = F, sep = "\t")
ROH_caseEnriched <- subset(ROH, caseEnriched == 1)
write.table(ROH_caseEnriched, file = "ROH_caseEnriched.txt", quote = F, row.names = F, sep = "\t")
