interface iicc #(parameter DWIDTH=16,localparam DBYTE=DWIDTH/8,parameter SIM=0)(
);

igticc #(.DWIDTH(DWIDTH)) gticc ();
wire [DWIDTH-1:0] txdata;
wire [DBYTE-1:0] txcharisk;
wire stbtxdata;
reg [DWIDTH-1:0] rxdata=0;
assign gticc.rxuserrdy=1'b1;
assign gticc.txuserrdy=1'b1;
wire resetdone=gticc.resetdone;

reg [47:0] txusrclkcnt=0;
wire [47:0] txcnt;
reg [47:0] rxcnt=0;
//reg [47:0] txcnt_corr=0;
//reg [47:0] txusrclkcnt_corr=0;
reg signed [47:0] corr48=0;
reg signed [3:0] corr4=0;
reg [1:0] corr250=0;
reg [1:0] corr1G=0;
reg phhalf=0;
reg [12:0] phcorr=0;
reg [47:0] txusrclkcnt_x=0;
wire [15:0] rxphdmtd;
wire [15:0] txphdmtd;
always @(posedge gticc.txusrclk) begin
	txusrclkcnt<=gticc.txreset ? 0 : txusrclkcnt+1;
end
always @(posedge gticc.rxusrclk) begin
	txusrclkcnt_x<=txusrclkcnt;
	rxcnt<=txcnt;
end

wire [DWIDTH-1:0] txdata_w;
reg [DBYTE-1:0] txcharisk_r=0;
reg [DWIDTH-1:0] rxdata_x=0;
reg [DBYTE-1:0] rxcharisk_x=0;
reg [63:0] t1tx=0;
reg [63:0] t2tx=0;
reg [63:0] t3tx=0;
reg [63:0] t4tx=0;

wire [63:0] t1rx;
reg [7:0] t1rxarray [7:0];
reg [7:0] t2rxarray [7:0];
reg [7:0] t3rxarray [7:0];
reg [7:0] t4rxarray [7:0];
wire [63:0] t2rx;
wire [63:0] t3rx;
wire [63:0] t4rx;
wire master;
wire [63:0] t1= master ? t1tx : t1rx;
wire [63:0] t2= master ? t2rx : t2tx;
wire [63:0] t3= master ? t3rx : t3tx;
wire [63:0] t4= master ? t4tx : t4rx;
wire [47:0] c1;
wire [47:0] c2;
wire [47:0] c3;
wire [47:0] c4;
wire [15:0] p1;
wire [15:0] p2;
wire [15:0] p3;
wire [15:0] p4;
assign {c1,p1}=t1;
assign {c2,p2}=t2;
assign {c3,p3}=t3;
assign {c4,p4}=t4;

assign t1rx={t1rxarray[0],t1rxarray[1],t1rxarray[2],t1rxarray[3],t1rxarray[4],t1rxarray[5],t1rxarray[6],t1rxarray[7]};
assign t2rx={t2rxarray[0],t2rxarray[1],t2rxarray[2],t2rxarray[3],t2rxarray[4],t2rxarray[5],t2rxarray[6],t2rxarray[7]};
assign t3rx={t3rxarray[0],t3rxarray[1],t3rxarray[2],t3rxarray[3],t3rxarray[4],t3rxarray[5],t3rxarray[6],t3rxarray[7]};
assign t4rx={t4rxarray[0],t4rxarray[1],t4rxarray[2],t4rxarray[3],t4rxarray[4],t4rxarray[5],t4rxarray[6],t4rxarray[7]};
reg [64:0] tdiff2=0;
reg [47:0] cdiff2=0;
reg [15:0] pdiff2=0;
reg stb_t1rx=0;
reg stb_t1tx=0;
reg alignrequest=0;
reg alignrequest_x=0;
reg rxbyterealign_x=0;
reg rxbyteisaligned_x=0;
reg txstb_r=0;
wire [DWIDTH-1:0] palignchar=16'h00bc;//{{(DWIDTH-8){1'b0}},8'hbc};
wire [DBYTE-1:0] paligncharisk=2'b01;//{{(DBYTE-1){1'b0}},1'b1};
//wire [DWIDTH-1:0] palignreq=16'h01bc;
//wire [DBYTE-1:0] palignreqisk=2'b01;
reg [4:0] actiontx=0;
wire  [2:0] indextx;
reg [7:0] txdata8=0;
wire txclk=gticc.txusrclk;
wire rxclk=gticc.rxusrclk;
reg [55:0] tsyncsr=0;
always @(posedge txclk) begin
	alignrequest_x<=alignrequest;
	rxbyterealign_x<=gticc.rxbyterealign;
end
reg [DWIDTH-1:0] rxchar_x=0;
reg [DBYTE-1:0] rxcharisk_xd=0;
reg [DWIDTH-1:0] rxchar_xd=0;
always @(posedge txclk) begin
	rxdata_x<=gticc.rxdata;
	rxcharisk_x<=gticc.rxcharisk;
	rxcharisk_xd<=rxcharisk_x;
	rxchar_xd<=rxdata_x;
	rxbyteisaligned_x<=gticc.rxbyteisaligned;
	if (|rxcharisk_x) begin
		if (rxcharisk_xd==2'b01 & rxchar_xd==16'h01bc) begin
			alignrequest<=1'b1;
		end
		else begin
			alignrequest<=1'b0;
		end

	end
	else begin
		alignrequest<=1'b0;
		rxdata<=rxdata_x;
/*		case (rxdata[15:8])
			8'h1: begin t1rx[8*8-1-:8] <= rxdata[7:0]; t2tx<={txusrclkcnt_x,rxphdmtd}; stb_t1rx<=1'b1; end
			8'h2: begin t1rx[7*8-1-:8] <= rxdata[7:0]; stb_t1rx<=1'b0; end
			8'h3: begin t1rx[6*8-1-:8] <= rxdata[7:0]; stb_t1rx<=1'b0; end
			8'h4: begin t1rx[5*8-1-:8] <= rxdata[7:0]; stb_t1rx<=1'b0; end
			8'h5: begin t1rx[4*8-1-:8] <= rxdata[7:0]; stb_t1rx<=1'b0; end
			8'h6: begin t1rx[3*8-1-:8] <= rxdata[7:0]; stb_t1rx<=1'b0; end
			8'h7: begin t1rx[2*8-1-:8] <= rxdata[7:0]; stb_t1rx<=1'b0; end
			8'h8: begin t1rx[1*8-1-:8] <= rxdata[7:0]; stb_t1rx<=1'b0; end
		endcase
		*/
	end
end
always @(posedge txclk) begin
end
reg [7:0] gott1=0;
reg [7:0] gott2=0;
reg [7:0] gott3=0;
reg [7:0] gott4=0;
enum {SYNCIDLE
,SYNCT1
,SYNCT2
,SYNCT3
,SYNCT4
,WAITT1
,WAITT2
,WAITT3
,WAITT4
,SYNCTO
,SYNCCALC
} syncstate=SYNCIDLE,syncnext=SYNCIDLE;
//reg [3:0] syncstate=SYNCIDLE;
reg [3:0] dbsyncstate=SYNCIDLE;
//reg [3:0] syncnext=SYNCIDLE;
reg [3:0] dbsyncnext=SYNCIDLE;
reg [3:0] syncnext_d=SYNCIDLE;
reg [31:0] synccnt=0;
assign indextx=synccnt[2:0];
reg stbsyncdiff=0;
wire syncreset=|{~gticc.resetdone,~gticc.rxbyteisaligned,gticc.txreset,gticc.rxreset,gticc.reset};
always @(posedge txclk) begin
	dbsyncnext<=syncnext;
	dbsyncstate<=syncstate;
	if (syncreset) begin
		syncstate<=SYNCIDLE;
	end
	else begin
		syncstate<=syncnext;
		synccnt<=(syncstate==syncnext) & (syncstate!=SYNCIDLE) ? synccnt+1 : 0;
	end
end
//wire synctrig=SIM ? ~|txusrclkcnt[8:0] : ~|txusrclkcnt[25:0];
wire synctrig=SIM ? ~|txcnt[8:0] : ~|txcnt[25:0];
always @(*) begin
	if (syncreset) begin
		syncnext=SYNCIDLE;
	end
	else begin
		case (syncstate)
			SYNCIDLE: syncnext = master ? synctrig  ? SYNCT1 : SYNCIDLE : WAITT1;
			SYNCT1: syncnext = synccnt==7 ? WAITT2 : SYNCT1;
			WAITT2: syncnext = synctrig ? SYNCTO : &gott2 ? WAITT3 : WAITT2;
			WAITT3: syncnext = synctrig ? SYNCTO : &gott3 ? SYNCT4 : WAITT3;
			SYNCT4: syncnext = synccnt==7 ? SYNCCALC : SYNCT4;

			WAITT1: syncnext = &gott1 ? SYNCT2 : WAITT1;
			SYNCT2: syncnext = synccnt==7 ? SYNCT3 : SYNCT2;
			SYNCT3: syncnext = synccnt==7 ? WAITT4 : SYNCT3;
			WAITT4: syncnext = synctrig ? SYNCTO : &gott4 ? SYNCCALC : WAITT4;
			SYNCCALC: syncnext = SYNCIDLE;

			SYNCTO: syncnext = SYNCIDLE;
		endcase
	end
end
wire [4:0] action=rxdata[15:11];
wire [2:0] index=rxdata[10:8];
assign indextx=synccnt[2:0];
reg usecorr=0;
//wire [63:0] cnt64rx= (~master & usecorr )? {txusrclkcnt_corr,rxphdmtd}:{txusrclkcnt,rxphdmtd};
//wire [63:0] cnt64tx= (~master & usecorr )? {txusrclkcnt_corr,txphdmtd}:{txusrclkcnt,txphdmtd};
wire [63:0] cnt64rx= {txcnt,rxphdmtd};// (~master & usecorr )? {txcnt_corr,rxphdmtd}:{txcnt,rxphdmtd};
//wire [63:0] cnt64tx= {txcnt,txphdmtd};//(~master & usecorr )? {txcnt_corr,txphdmtd}:{txcnt,txphdmtd};
wire [63:0] cnt64tx= {txcnt,txphdmtd};//(~master & usecorr )? {txcnt_corr,txphdmtd}:{txcnt,txphdmtd};
//wire [63:0] cnt64={txusrclkcnt,rxphdmtd};
reg first1=0;
reg first2=0;
reg first3=0;
reg first4=0;
reg txdata_r=0;
reg synctx=0;
reg t2done=0;
reg t3done=0;
reg t4done=0;
reg t1done=0;
reg aligntx=0;
reg [15:0] phdiff=0;
always @(posedge txclk) begin
	if (syncreset) begin
		usecorr<=1'b0;
	end
	else begin
		case (syncnext)
			SYNCIDLE:begin
				{gott1,gott2,gott3,gott4}<=0;
				{first4,first3,first2,first1}<=4'hf;
				txstb_r<=stbtxdata;
				stbsyncdiff<=1'b0;
				synctx<=1'b0;
				aligntx<=1'b0;
			end
			SYNCT1: begin
				first1<=1'b0;
				if (first1)
					t1tx<=cnt64tx;
					//t1tx<=cnt64rx;
				actiontx<=5'h1;
				//{txdata8,tsyncsr}<=first1 ? cnt64rx : {tsyncsr,8'h0};
				{txdata8,tsyncsr}<=first1 ? cnt64tx : {tsyncsr,8'h0};
				txstb_r<=1'b1;
				synctx<=1'b1;
				txcharisk_r<=0;
			end
			SYNCT2: begin
				first2<=1'b0;
				actiontx<=5'h2;
				{txdata8,tsyncsr}<=first2 ? t2tx : {tsyncsr,8'h0};
				txstb_r<=1'b1;
				synctx<=1'b1;
				txcharisk_r<=0;
			end
			SYNCT3: begin
				first3<=1'b0;
				if (first3)
					t3tx<=cnt64tx;
					//t3tx<=cnt64rx;
				actiontx<=5'h3;
				{txdata8,tsyncsr}<=first3 ? cnt64tx : {tsyncsr,8'h0};
				//{txdata8,tsyncsr}<=first3 ? cnt64rx : {tsyncsr,8'h0};
				txstb_r<=1'b1;
				synctx<=1'b1;
				txcharisk_r<=0;
			end
			SYNCT4: begin
				first4<=1'b0;
				actiontx<=5'h4;
				{txdata8,tsyncsr}<=first4 ? t4tx : {tsyncsr,8'h0};
				txstb_r<=1'b1;
				synctx<=1'b1;
				txcharisk_r<=0;
			end
			WAITT1: begin
				{first4,first3,first2,first1}<=4'hf;
				if (action==5'h1) begin
					if (~|gott1) begin
						t2tx<=cnt64rx;
						t2done<=1'b1;
					end
					gott1[index]<=1'b1;
					t1rxarray[index]=rxdata[7:0];
				end
				else begin
					gott1<=0;
				end
				synctx<=1'b0;
				txstb_r<=stbtxdata;
			end
			WAITT2: begin
				txstb_r<=stbtxdata;
				{first4,first3,first2,first1}<=4'hf;
				if (action==5'h2) begin
					gott2[index]<=1'b1;
					t2rxarray[index]=rxdata[7:0];
				end
				synctx<=1'b0;
			end
			WAITT3: begin
				txstb_r<=stbtxdata;
				synctx<=1'b0;
				{first4,first3,first2,first1}<=4'hf;
				if (action==5'h3) begin
					if (~|gott3) begin
						t4tx<=cnt64rx;
						t4done<=1'b1;
					end
					gott3[index]<=1'b1;
					t3rxarray[index]=rxdata[7:0];
				end
			end
			WAITT4: begin
				txstb_r<=stbtxdata;
				synctx<=1'b0;
				{first4,first3,first2,first1}<=4'hf;
				if (action==5'h4) begin
					gott4[index]<=1'b1;
					t4rxarray[index]=rxdata[7:0];
				end
			end
			SYNCCALC: begin
				{first4,first3,first2,first1}<=4'hf;
				//tdiff2<= master ? ((t4tx-t1tx)-(t3rx-t2rx)):((t4rx-t1rx)-(t3tx-t2tx));
				tdiff2<= master ? (t4tx+t1tx-t3rx-t2rx):(t4rx+t1rx-t3tx-t2tx);
				cdiff2<= c4+c1-c3-c2;//(t4tx+t1tx-t3rx-t2rx):(t4rx+t1rx-t3tx-t2tx);
				pdiff2<= p4+p1-p3-p2;//(t4tx+t1tx-t3rx-t2rx):(t4rx+t1rx-t3tx-t2tx);
				txstb_r<=1'b0;
				stbsyncdiff<=1'b1;
				synctx<=1'b0;
				usecorr<=1'b1;
				aligntx<=1'b1;
			end
		endcase
	end
end
reg stb_corr=0;
wire [15:0] pdiff=(pdiff2>>1);//+pdiff2[0];
always@(posedge txclk) begin
	if (stbsyncdiff) begin
		corr48<=corr48+(cdiff2>>>1);//+cdiff2[0];//(tdiff2>>>17)+tdiff2[16];
		corr4<=pdiff[13:10];
		corr250<=pdiff[13:12];
		corr1G<=pdiff[11:10];
		phdiff<={pdiff[13:0],2'b0};
		phhalf<=pdiff[9];
		phcorr<=pdiff[8:0];
	end
	stb_corr<=stbsyncdiff;
	//txusrclkcnt_corr<=txusrclkcnt+corr48;
	//txcnt_corr<=txcnt+corr48;
//	phdiff<=(tdiff2[14:1]+tdiff2[0]);
end

//assign gticc.txdata= ~rxbyteisaligned_x ? palignreq : alignrequest_x ? palignchar : txstb_r ? synctx ? {actiontx,indextx,txdata8} : txdata : palignchar;
//assign gticc.txcharisk= ~rxbyteisaligned_x ? palignreqisk :  alignrequest_x ?  paligncharisk : txstb_r ? 0: paligncharisk;
assign gticc.txdata=    txstb_r ? (synctx ? {actiontx,indextx,txdata8} : (aligntx ? palignchar    : txdata      )) : palignchar   ;
assign gticc.txcharisk= txstb_r ? (synctx ? 0                          : (aligntx ? paligncharisk : txcharisk   )) : paligncharisk;

/*modport cfg (input rxphdmtd
,output
);
*/
endinterface

interface igticc #(parameter DWIDTH=16,localparam DBYTE=DWIDTH/8)(
);
wire rxusrclk;
wire [DBYTE-1:0]rxcharisk;
wire [DBYTE-1:0] rxdisperr;
wire [DBYTE-1:0] rxnotintable;
wire [DWIDTH-1:0] rxdata;
wire rxuserrdy;
wire rxoutclkfabric;
wire rxbyteisaligned;
wire rxbyterealign;

wire txusrclk;
wire [DBYTE-1:0]txcharisk;
wire txuserrdy;
wire [DWIDTH-1:0] txdata;
wire [DWIDTH-1:0] txdataout;
wire [DBYTE-1:0]txchariskout;
wire reset;
wire txreset;
wire rxreset;
wire resetdone;

modport gt (inout reset
,input rxuserrdy,txuserrdy,txdata,txcharisk
,output txusrclk,rxusrclk,rxdata,rxdisperr,rxnotintable,rxcharisk,rxbyteisaligned,rxbyterealign,txdataout,txchariskout,resetdone,rxreset,txreset
);
endinterface

