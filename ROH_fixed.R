library(data.table)
library(dplyr)
library(tidyr)
library(plyr)
setwd("~/Desktop/")
ROH <- fread("ROH_tocheck.txt", header = T)
ROH$case <- ldply(strsplit(as.character(ROH$case_to_control), split = ":"))[[1]]
ROH$control <- ldply(strsplit(as.character(ROH$case_to_control), split = ":"))[[2]]
ROH$NcasesWithROHs <- as.numeric(ROH$case)
ROH$NcontrolsWithROHs <- as.numeric(ROH$control)
ROH$caseN <- as.numeric(ROH$case)
ROH$controlN <- as.numeric(ROH$control)
ROH$combinedN <- ROH$caseN + ROH$controlN
ROH$P <- NA
for(i in 1:length(ROH$case))
{
  thisP <- prop.test(x = c(ROH$NcasesWithROHs[i], ROH$NcontrolsWithROHs[i]), n = c(1070.00, 4711.0)) # note the last part should be the total N for cases and then the total N for controls.
  ROH$P[i] <- thisP$p.value
}
ROH$total_ROH_count <- ROH$caseN + ROH$controlN
ROH$propCases <- ROH$caseN/1070.00
ROH$propControls <- ROH$controlN/4711.0
ROH$caseEnriched <- ifelse(ROH$propCases > ROH$propControls, 1, 0)
ROH_subsetted <- subset(ROH, total_ROH_count >= 1 & caseEnriched == 1)
Ngenes <- length(ROH_subsetted$PD)
ROH_subsetted$passMultiTest <- ifelse(ROH_subsetted$P <= (0.05/Ngenes),1,0)
subset(ROH_subsetted, passMultiTest == 1)
write.table(ROH_subsetted, file = "ROH_subsetted.txt", quote = F, row.names = F, sep = "\t")
ROH_caseEnriched <- subset(ROH, caseEnriched == 1)
write.table(ROH_caseEnriched, file = "ROH_caseEnriched.txt", quote = F, row.names = F, sep = "\t")
