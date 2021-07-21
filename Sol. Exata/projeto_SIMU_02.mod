reset;
set FABRICANTE;
set GRUPO;   
set DOSE;  
set UNIDADE;
  

param peso {DOSE,GRUPO} >= 0;  

param estoque {FABRICANTE} >= 0;


param demanda {GRUPO,DOSE,UNIDADE} >= 0;  #cria matriz tridimensional demanda
var DISTRIBUIDA {GRUPO,DOSE,UNIDADE} >= 0;  #cria matriz tridimensional distribuida
param fabricanteDose  {FABRICANTE,DOSE} binary; #cria matriz para definir qual dose por ser utilizada por fabricante

#A fun��o objetivo busca maximizar a quantidade de vacinas distribu�das por unidade, dose e grupo priorit�rio, multiplicado pela import�ncia da dose para o grupo priorit�rio. Exemplo segunda dose da vacina da Astrazeneca para profissionais de sa�de.
maximize Total_Priorizado:   sum {k in GRUPO, j in DOSE, i in UNIDADE}  peso[j,k] * DISTRIBUIDA[k,j,i];


#Quantidade distribu�da n�o pode ultrapassar a necess�ria para cada unidade, dose e grupo priorit�rio. 
subject to Demanda {j in DOSE, k in GRUPO, i in UNIDADE}: DISTRIBUIDA[k,j,i] <= demanda[k,j,i];
#Quantidade distribu�da n�o pode ultrapassar a quantidade de vacinas existentes no estoque, de acordo com seu fabricante
subject to Estoque: sum {k in GRUPO, j in DOSE, i in UNIDADE} DISTRIBUIDA[k,j,i] <= sum{f in FABRICANTE} estoque[f];


   
data projeto_SIMU_02.dat; 
option solver cplex;
option times 1;
option cplex_options 'timing 1';


solve;
display {i in UNIDADE}: {k in GRUPO, j in DOSE} DISTRIBUIDA[k,j,i];
