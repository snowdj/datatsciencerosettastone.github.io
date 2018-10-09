/*******************************************************
Linear Models of Panel Data Using Greene's U.S. airline data (2000)
Created on June 21, 2004
Last modified on 09/05/2009; September 30, 2005

Author: Hun Myoung Park
********************************************************/



OPTIONS NODATE NONUMBER;
LIBNAME masil 'c:\temp\sas';




/******************************************
IT firm R&D expenditure (OECD 2002)
******************************************/

TITLE "IT firm R and D expenditure (OECD 2002)";
TITLE2 "LSDV 1 w/ d1";
PROC REG DATA=masil.rnd2002;
   MODEL rnd = income d1;
RUN;

TITLE2 "LSDV 1 w/ d2";
PROC REG DATA=masil.rnd2002;
   MODEL rnd = income d2;
RUN;

TITLE2 "LSDV 2";
PROC REG DATA=masil.rnd2002;
   MODEL rnd = income d1 d2 /NOINT;
RUN;

TITLE2 "LSDV 3";
PROC REG DATA=masil.rnd2002;
   MODEL rnd = income d1 d2;
   RESTRICT d1 + d2 = 0;
RUN;

TITLE2 "LSDV 1: PROC GLM";
PROC GLM DATA=masil.rnd2002;
   MODEL rnd = income d2 /SOLUTION;
RUN;

TITLE2 "LSDV 1: PROC MIXED";
PROC MIXED DATA=masil.rnd2002;
   MODEL rnd = income d2 /SOLUTION;
RUN;




/******************************************
U.S. Airline Cost Data (Greene 2003)
******************************************/

TITLE "U.S. Airline Cost Data (Greene 2003)";
TITLE2 "Pooled OLS regression";
PROC REG DATA=masil.airline; 
   MODEL cost = output fuel load; 
RUN;





/*************************************************************************/
/* One-way group effect model */

TITLE2 "One-way (Individual-specifict) Effect: LSDV 1";
PROC REG DATA=masil.airline; 
   MODEL cost = g1-g5 output fuel load; 
RUN;

TITLE2 "One-way (Individual-specifict) Effect: LSDV 2";
PROC REG DATA=masil.airline;  
   MODEL cost = g1-g6 output fuel load /NOINT; 
RUN;

TITLE2 "One-way (Individual-specifict) Effect: LSDV 3";
PROC REG DATA=masil.airline;  
   MODEL cost = g1-g6 output fuel load; 
   RESTRICT g1 + g2 + g3 + g4 + g5 + g6 = 0;
RUN;

TITLE2 "One-way (Individual-specifict) Effect: Hypothesis Test";
PROC REG DATA=masil.airline; 
   MODEL cost = g1-g5 output fuel load; 
   TEST g1 = g2 = g3 = g4 = g5 = 0;
RUN;



PROC SORT DATA=masil.airline;      
   BY airline year; 
RUN;

TITLE2 "One-way Group (Individual-specifict) Effect: Within Effect";
PROC TSCSREG DATA=masil.airline;
   ID airline year;
   MODEL cost = output fuel load /FIXONE;
RUN;

TITLE2 "One-way Group (Individual-specifict) Effect: Within Effect";
PROC PANEL DATA=masil.airline;
   ID airline year;
   MODEL cost = output fuel load /FIXONE;
RUN;

TITLE2 "Between Group Effect Model";
PROC PANEL DATA=masil.airline; 
   ID airline year;
   MODEL cost = output fuel load /BTWNG;
RUN;





/*************************************************************************/
/* One-way time effect model */

TITLE2 "One-way Time Effect: LSDV 1";
PROC REG DATA=masil.airline; 
   MODEL cost = t1-t14 output fuel load; 
RUN;

TITLE2 "One-way Time Effect: LSDV 2";
PROC REG DATA=masil.airline;  
   MODEL cost = t1-t15 output fuel load /NOINT; 
RUN;

TITLE2 "One-way Time Effect: LSDV 3";
PROC REG DATA=masil.airline;  
   MODEL cost = t1-t15 output fuel load; 
   RESTRICT t1+t2+t3+t4+t5+t6+t7+t8+t9+t10+t11+t12+t13+t14+t15=0;
RUN;

TITLE2 "One-way Time Effect: Hypothesis Test";
PROC REG DATA=masil.airline; 
   MODEL cost = t1-t14 output fuel load; 
   TEST t1=t2=t3=t4=t5=t6=t7=t8=t9=t10=t11=t12=t13=t14=0;
RUN;



PROC SORT DATA=masil.airline;      
   BY year airline; 
RUN;

TITLE2 "One-way Time Effect: Within Effect";
PROC TSCSREG DATA=masil.airline;
   ID year airline;
   MODEL cost = output fuel load /FIXONE;
RUN;

TITLE2 "One-way Time Effect: Within Effect";
PROC PANEL DATA=masil.airline;
   ID year airline;
   MODEL cost = output fuel load /FIXONE;
RUN;

TITLE2 "Between Time Effect Model: /BTWNG";
PROC PANEL DATA=masil.airline; /*  */
   ID year airline;
   MODEL cost = output fuel load /BTWNG;
RUN;

PROC SORT DATA=masil.airline;      
   BY airline year; 
RUN;

TITLE2 "Between Time Effect Model: /BTWNT";
PROC PANEL DATA=masil.airline;
   ID airline year;
   MODEL cost = output fuel load /BTWNT;
RUN;





/*************************************************************************/
/* Two-way Fixed Effect Model */

TITLE2 "Two-way Fixed Effect Model: LSDV1 + LSDV1";
PROC REG DATA=masil.airline; 
   MODEL cost = g1-g5 t1-t14 output fuel load; 
RUN;

TITLE2 "Two-way Fixed Effect Model: LSDV1 + LSDV2";
PROC REG DATA=masil.airline;  
   MODEL cost = g1-g5 t1-t15 output fuel load /NOINT;
   MODEL cost = g1-g6 t1-t14 output fuel load /NOINT; 
RUN;

TITLE2 "Two-way Fixed Effect Model: LSDV1 + LSDV3";
PROC REG DATA=masil.airline;  
   MODEL cost = g1-g6 t1-t14 output fuel load; 
   RESTRICT g1 + g2 + g3 + g4 + g5 + g6 = 0;
RUN;

TITLE2 "Two-way Fixed Effect Model: LSDV1 + LSDV2";
PROC REG DATA=masil.airline;  
   MODEL cost = g1-g5 t1-t15 output fuel load; 
   RESTRICT t1+t2+t3+t4+t5+t6+t7+t8+t9+t10+t11+t12+t13+t14+t15=0;
RUN;

TITLE2 "Two-way Fixed Effect Model: LSDV2 + LSDV3";
PROC REG DATA=masil.airline;
   MODEL cost = g1-g6 t1-t15 output fuel load /NOINT; 
   RESTRICT g1 + g2 + g3 + g4 + g5 + g6 = 0;
RUN;

TITLE2 "Two-way Fixed Effect Model: LSDV2 + LSDV3";
PROC REG DATA=masil.airline; 
   MODEL cost = g1-g6 t1-t15 output fuel load /NOINT; 
   RESTRICT t1+t2+t3+t4+t5+t6+t7+t8+t9+t10+t11+t12+t13+t14+t15=0;
RUN;

TITLE2 "Two-way Fixed Effect Model: LSDV 3 + LSDV3";
PROC REG DATA=masil.airline;  /* LSDV3 */
   MODEL cost = g1-g6 t1-t15 output fuel load; 
   RESTRICT g1 + g2 + g3 + g4 + g5 + g6 = 0;
   RESTRICT t1+t2+t3+t4+t5+t6+t7+t8+t9+t10+t11+t12+t13+t14+t15=0;
RUN;

TITLE2 "Two-way Fixed Effect Model: Hypothesis Test";
PROC REG DATA=masil.airline; 
   MODEL cost = g1-g5 t1-t14 output fuel load; 
   TEST g1=g2=g3=g4=g5=t1=t2=t3=t4=t5=t6=t7=t8=t9=t10=t11=t12=t13=t14=0;
RUN;


PROC SORT DATA=masil.airline;      
   BY airline year; 
RUN;

TITLE2 "Two-way Fixed Effect Model: Within Effect";
PROC TSCSREG DATA=masil.airline;
   ID airline year;
   MODEL cost = output fuel load /FIXTWO;
RUN;

TITLE2 "Two-way Fixed Effect Model: Within Effect";
PROC PANEL DATA=masil.airline;
   ID airline year;
   MODEL cost = output fuel load /FIXTWO;
RUN;





/*************************************************************************/
/* Random Effect Model */

TITLE2 "One-way Random Group Effect";
PROC SORT DATA=masil.airline;      
   BY airline year; 
RUN;

PROC TSCSREG DATA=masil.airline;
   ID airline year;
   MODEL cost = output fuel load /RANONE;
RUN;

TITLE2 "One-way Random Group Effect: VCOMP=WK";
PROC PANEL DATA=masil.airline;
   ID airline year;
   MODEL cost = output fuel load /RANONE BP VCOMP=WK;
RUN;

TITLE2 "One-way Random Group Effect: VCOMP=WH";
PROC PANEL DATA=masil.airline;
   ID airline year;
   MODEL cost = output fuel load /RANONE BP VCOMP=WH;
RUN;

TITLE2 "One-way Random Group Effect: MIXED";
PROC MIXED DATA=masil.airline ;
	CLASS airline;
    MODEL cost = output fuel load /SOLUTION;
	RANDOM INTERCEPT / SUBJECT=airline TYPE=UN SOLUTION;
RUN;

TITLE2 "One-way Random Group Effect: MIXED w/ METHOD=ML";
PROC MIXED DATA=masil.airline METHOD=ML;
	CLASS airline;
    MODEL cost = output fuel load /SOLUTION;
	RANDOM INTERCEPT / SUBJECT=airline TYPE=UN SOLUTION;
RUN;



TITLE2 "One-way Random Time Effect";
PROC SORT DATA=masil.airline;      
   BY year airline; 
RUN;

PROC TSCSREG DATA=masil.airline;
   ID year airline;
   MODEL cost = output fuel load /RANONE;
RUN;

PROC PANEL DATA=masil.airline;
   ID year airline;
   MODEL cost = output fuel load /RANONE BP;
RUN;

TITLE2 "One-way Random Time Effect: VCOMP=WK";
PROC PANEL DATA=masil.airline;
   ID year airline;
   MODEL cost = output fuel load /RANONE BP VCOMP=WK;
RUN;

TITLE2 "One-way Random Time Effect: VCOMP=WH";
PROC PANEL DATA=masil.airline;
   ID year airline;
   MODEL cost = output fuel load /RANONE BP VCOMP=WH;
RUN;

TITLE2 "One-way Random Time Effect: MIXED";
PROC MIXED DATA=masil.airline;
	CLASS year;
	MODEL cost = output fuel load /SOLUTION;
	RANDOM INTERCEPT / SUBJECT=year TYPE=UN;
RUN;

TITLE2 "One-way Random Time Effect: MIXED";
PROC MIXED DATA=masil.airline ;
	CLASS year;
    MODEL cost = output fuel load /SOLUTION;
	RANDOM INTERCEPT / SUBJECT=year TYPE=UN SOLUTION;
RUN;





/*************************************************************************/
/* Two-way random time effect */

PROC SORT DATA=masil.airline;      
   BY airline year; 
RUN;

TITLE2 "Two-way Random Effect Model";
PROC TSCSREG DATA=masil.airline;
   ID airline year;
   MODEL cost = output fuel load /RANTWO;
RUN;

PROC PANEL DATA=masil.airline;
   ID airline year;
   MODEL cost = output fuel load /RANTWO BP2;
RUN;

TITLE2 "Two-way Random Effect Model: /VCOMP=WH";
PROC PANEL DATA=masil.airline;
   ID airline year;
   MODEL cost = output fuel load /RANTWO BP2 VCOMP=WH;
RUN;

TITLE2 "Two-way Random Effect Model: /VCOMP=WK";
PROC PANEL DATA=masil.airline;
   ID airline year;
   MODEL cost = output fuel load /RANTWO BP2 VCOMP=WK;
RUN;






/*************************************************************************/
/* Poolability Test */

TITLE2 "Poolability Test: Group";
PROC SORT DATA=masil.airline;
    BY airline;
RUN;

PROC REG DATA=masil.airline;
   MODEL cost = output fuel load;
   BY airline;
RUN;


TITLE2 "Poolability Test: Time";
PROC SORT DATA=masil.airline;
    BY year;
RUN;

PROC REG DATA=masil.airline;
   MODEL cost = output fuel load;
   BY year;
RUN;
