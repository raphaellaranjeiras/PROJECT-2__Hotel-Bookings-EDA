![](hotel.png)

# Projeto 2: Hotel Bookings
  O conjunto de dados foi tirado do Kaggle e pode ser acessado [aqui](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). 
  
  ## Informações
- O conjunto de dados fala sobre as reservas de hotéis de dois tipos ao longo do tempo, sua composição e o status da reserva.
- O Data Set cobre o **período** de **julho de 2015** até **agosto de 2017**.
- **Reservas bem sucedidas**: Marcação e chegada no hotel.
- **Estadia**: Quantidade de dia(s)/tempo que se passou no hotel.
- Esse **projeto** foi **dividido em duas partes**, sendo essa composta de uma **Análise Exploratória (EDA) extensa**. 
- O **principal objetivo** dessa EDA foi desenvolver um maior entendimento do comportamento e características dos clientes, para que dessa forma possa favorecer a implementação de melhorias na oferta do serviço e no planejamento da administração. Além disso, conforme será visto no projeto seguinte, essa análise também tem como alvo uma maior familiarização com os dados para futura implementação de um modelo de machine learning.
- A análise exploratória foi divida em 3 partes por questões de conforto e facilitação da visualização da mesma. 
- **Todos** os **comentários** e **insights** pertinentes estão presentes ao longo dos **respectivos notebooks**.
- A **biblioteca** de visualização **utilizada** foi a ***Seaborn***, amplamente difundida.

- **Algumas** das perguntas que foram escolhidas para serem respondidas com a presente análise:
  - Quais foram os meses como maior e menor chegada de novos hóspedes?
  - Quais foram os meses como maior e menor estadias programadas?
  - Qual a nacionalidade dos hóspedes que mais frequentam os hotéis?
  - Como se comportam os clientes no que tange ao momento da realização da reserva e qual é/existe relação com os cancelamentos? 
  - Clientes repitidos tendem a realizar cancelamentos?
  - Quais os meses que tendem a ocorrer mais cancelamentos?
  - O segmento de mercado tem relação com a taxa de cancelamento?
  - Qual o canal de distribuição mais rentável?
  - Hóspedes de quais países tendem a dar mais renda aos hotéis?
  
### Variáveis presentes:

- *hotel* - Tipo do hotel (Hotel da Cidade ou Resort).
- *is_canceled* - Status da Reserva, se a mesma foi cancelada ou não. 0 -> Não cancelada, 1 -> Cancelada.
- *lead_time* - Número de dias decorridos entre a data de entrada da reserva no PMS (Property Management System) e a data de chegada no hotel.
- *arrival_date_year* - Ano de chegada no hotel.
- *arrival_date_month* - Mês de chegada no hotel.
- *arrival_date_week_number* - Número da semana de chegada no ano.
- *arrival_date_day_of_month* - Data do dia de chegada no hotel.
- *stays_in_weekend_nights* - Estadias aos finais de semana.
- *stays_in_week_nights* - Estadias nos dias de semana. 
- *adults* - Número de adultos que compõem a reserva. 
- *children* - Número de crianças que compõem a reserva. 
- *babies* - Número de bebês que compõem a reserva.
- *meal* - Tipo de refeição escolhida. As categorias são apresentadas em pacotes de refeição de hospitalidade padrão: Undefined/SC (Sem pacote de refeição), BB (Cama & Café da manhã), HB (Café da manhã e mais uma refeição, geralmente jantar) e Full Board (Café da manhã, almoço e jantar).
- *country* - País de origem. As categorias são representadas no formato da ISO 3155–3:2013.
- *market_segment* - Designação do segmento de mercado. Nas categorias, o termo TA significa Agentes de Viagens e o termo TO Operadores de Viagens. 
- *distribution_channel* - Canal de distribuição das reservas. O termo TA significa Agentes de Viagens e o termo TO Operadores de Viagens.
- *is_repeated_guest* - Indica se o hóspede é repedito ou não. 0 -> Não repetido, 1 -> repetido.
- *previous_cancellations* - Número de cancelamentos prévios feitos pelo cliente anteriormente à reserva atual. 
- *previous_bookings_not_canceled* - Número de não cancelamentos prévios feitos pelo cliente anteriormente à reserva atual. 
- *reserved_room_type* - Código do tipo do quarto reservado. É apresentado em forma de código por questões de anonimato. 
- *assigned_room_type* - Código do tipo do quarto que de fato foi atribuído à reserva. Está sujeito a ser diferente do quarto que foi reservado devido a problemas na operacionalização (por exemplo, overbooking) ou até mesmo por solicitação do cliente. É apresentado em código também por motivos de anonimato.   
- *booking_changes* - Número de alterações feitas na reserva desde o momento em que a mesma entrou no PMS até ao momento do check-in ou cancelamento.
- *deposit_type* - Tipo do depósito que o cliente fez (caso tenha feito) para garantir a reserva. No Deposit (Sem depósito), Refundable (Depósito foi feito com um valor inferior ao custo total) e Non Refund (Depósito foi feito com o valor total da estadia). 
- *agent* - ID da agência de viagem que fez a reserva. ID foi apresentado por razões de anonimato.
- *company* - ID da companhia ou entidade que fez a reserva ou se responsabilizou por paga-la. ID foi apresentado por razões de anonimato. 
- *days_in_waiting_list* - Número de dias que a reserva ficou na lista de espera antes de ser confirmada ao cliente.
- *customer_type* - Tipo de reserva, assumindo as seguintes categorias: Transient (quando a reserva não faz parte de um grupo ou contrato), Contract (quando a reserva tem algum tipo de contrato atrelado), Group (quando a reserva é associada a um grupo de pessoas) e Transient-party (quando a reserva é Transient mas é associada com pelo menos outra reseva Transiet).
- *adr* - Taxa diária média. É definida pela divisão da soma de todas as transações de hospedagem pelo número total de noites de hospedagem.
- *required_car_parking_spaces* - Número de vagas de estacionamento exigidas pelo cliente.
- *total_of_special_requests* - Número de pedidos especiais feitos pelo cliente (por exemplo, cama de solteiro ou andar alto).
- *reservation_status* - Último status da reserva. Canceled (Reserva cancelada pelo cliente), Check-Out (O cliente fez check-in mas já partiu) e No-Show (O cliente não fez check-in e informou o hotel sobre o motivo).
- *reservation_status_date* - Data em que o último status da reserva foi estabelecido.

---
#### 💡 Sinta-se livre para contribuir: correções, sugestões, novos insights, muito bem vindos! Valeu! :smiley:
