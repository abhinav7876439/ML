```mermaid
graph TD
    subgraph Active_Family
    A1[1: (0,1)] 
    A2[2: (1,1)] 
    A3[3: (2,1)] 
    A4[4: (3,1)]
    end

    subgraph Passive_Family
    P5[5: (0,0)] 
    P6[6: (1,0)] 
    P7[7: (2,0)] 
    P8[8: (3,0)]
    end
    
    A1 -->|Prob 1| A1
    A2 -->|Prob 1| A1
    A3 -->|Prob 1| A1
    A4 -->|Prob 1| A1
    P5 -->|Prob 1| A1
    P6 -->|Prob 1| A1
    P7 -->|Prob 1| A1
    P8 -->|Prob 1| A1
    
    style A1 fill:#f9f,stroke:#333
```