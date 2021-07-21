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

#A função objetivo busca maximizar a quantidade de vacinas distribuídas por unidade, dose e grupo prioritário, multiplicado pela importância da dose para o grupo prioritário. Exemplo segunda dose da vacina da Astrazeneca para profissionais de saúde.
maximize Total_Priorizado:   sum {k in GRUPO, j in DOSE, i in UNIDADE}  peso[j,k] * DISTRIBUIDA[k,j,i];


#Quantidade distribuída não pode ultrapassar a necessária para cada unidade, dose e grupo prioritário. 
subject to Demanda {j in DOSE, k in GRUPO, i in UNIDADE}: DISTRIBUIDA[k,j,i] <= demanda[k,j,i];
#Quantidade distribuída não pode ultrapassar a quantidade de vacinas existentes no estoque, de acordo com seu fabricante
subject to Estoque: sum {k in GRUPO, j in DOSE, i in UNIDADE} DISTRIBUIDA[k,j,i] <= sum{f in FABRICANTE} estoque[f];


   
data projeto_SIMU_02.dat; 
option solver cplex;
option times 1;
option cplex_options 'timing 1';


solve;
display {i in UNIDADE}: {k in GRUPO, j in DOSE} DISTRIBUIDA[k,j,i];
