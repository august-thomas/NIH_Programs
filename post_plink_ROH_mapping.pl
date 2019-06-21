#!/usr/bin/perl -w

#command input_overlap input_CON_UNICON input_gene output_CON_with_gene    

use strict;
use warnings;

my $input_overlap = $ARGV[0];
my $input_CON = $ARGV[1];
my $input_gene = $ARGV[2];
my $output_CON_with_gene = $ARGV[3];

open(INPUT_FILE_00, "$input_overlap") or
     die "Could not open input $input_overlap\n";
open(INPUT_FILE_0, "$input_CON") or
     die "Could not open input $input_CON\n";
open(INPUT_FILE, "$input_gene") or
      die "Could not open input $input_gene\n";
open(OUTPUT_FILE, ">$output_CON_with_gene") or
      die "Could not open output file $output_CON_with_gene\n";

my @overlaps = <INPUT_FILE_00>;
shift @overlaps;
my %hit_cases;
my %hit_controls;
foreach my $overlap (@overlaps) {
        $overlap =~ s/^\s+//;
        chomp $overlap;
   if ( $overlap ) {
        (my $POOL,  my $FID, my $IID, my $PHE, my $CHR, my $SNP1, my $SNP2, my $BP1, my $BP2, my $KB, my $NSNP, my $NSIM, my $GRP) = split /\s+/, $overlap;
          $POOL =~ s/\s//g;
         if ($PHE == 1) {
                $hit_controls{$POOL} .= $IID." " ;
         }
         if ($PHE == 2) {
                $hit_cases{$POOL} .= $IID." ";
         }
   }
}


my @CONs = <INPUT_FILE_0>;
my @genes = <INPUT_FILE>;
my %genes;
foreach my $gene (@genes) {
	chomp $gene;
	(my $pos, my $gene_name) = split /\t/, $gene;
	$gene_name =~ s/\s//g; 
	$genes{$pos} = $gene_name;	
}

shift @CONs;
print OUTPUT_FILE "POOL\tnum_subjects\tcase_to_control\tratio\tCHR\tSNP1\tSNP2\tBP1\tBP2\tKB\tNSNP\tPD gene\tcases_IDs\tcontrol_IDs\n";
my %hit_list;
foreach my $CON (@CONs) {
 	chomp $CON;
	$CON =~ s/^\s+//;
        (my $POOL,  my $FID, my $IID, my $case_to_control, my $CHR, my $SNP1, my $SNP2, my $BP1, my $BP2, my $KB, my $NSNP, my $NSIM, my $GRP) = split /\s+/, $CON;
	foreach my $id (sort keys %genes) {
	     (my $g_chr, my $range) = split /\:/, $id;
             (my $g_pos1, my $g_pos2) = split /-/, $range;
	     $g_chr =~ s/chr//g; 
	     if ($g_chr == $CHR) {
		if (($g_pos1 <= $BP2 && $g_pos1 >= $BP1) || ($g_pos2 <= $BP2 && $g_pos2 >= $BP1) || (($BP1 <= $g_pos2 && $BP1 >= $g_pos1) && ($BP2 <= $g_pos2 && $BP2 >= $g_pos1) ) ) {
		   $hit_list{$CON} .= $genes{$id}." ";	 
		print "PD gene: $genes{$id}\n";
		} 
             }
        }
}
foreach my $hit (sort keys %hit_list) {
	(my $POOL,  my $FID, my $IID, my $case_to_control, my $CHR, my $SNP1, my $SNP2, my $BP1, my $BP2, my $KB, my $NSNP, my $NSIM, my $GRP) = split /\s+/, $hit;
	(my $num_cases, my $num_controls) = split /\:/, $case_to_control;
        my $ratio = $case_to_control; 
        if ($num_controls != 0) {
	 $ratio = sprintf "%.1f", $num_cases/$num_controls; 
        }
     my $list_cases = 0;
     my $list_controls = 0;
     if ($hit_cases{$POOL}) {
        $list_cases = $hit_cases{$POOL};
     }
     if ($hit_controls{$POOL}) {
        $list_controls =  $hit_controls{$POOL};
     }
     print OUTPUT_FILE $POOL, "\t", $IID, "\t\"", $case_to_control, "\"\t", $ratio, "\tchr", $CHR, "\t", $SNP1, "\t", $SNP2, "\t", $BP1, "\t", $BP2, "\t", $KB, "\t", $NSNP, "\t",  $hit_list{$hit},  "\t",  $list_cases, "\t", $list_controls, "\n";
}


close INPUT_FILE_00;
close INPUT_FILE_0;
close INPUT_FILE;
close OUTPUT_FILE;

exit;

