#include <iostream>
using namespace std;

void BestV(int c,int *w, int *v, int n, int **m, int **b);
void BestS(int **b, int *w, int *x, int n, int c);

int main()
{
	int c,n,i,j;
	int *w,*p,*x;
	int **m,**b;
	
	freopen("0-1in2.txt","r",stdin);
	freopen("0-1out2.txt","w",stdout);
	
	cin>>c>>n;
	
	w=new int[n+1], p=new int[n+1];
	for(i=1;i<=n;i++) cin>>w[i];
	for(i=1;i<=n;i++) cin>>p[i];
	
	m=new int *[n+1], b=new int *[n+1];
	for(i=1;i<=n;i++)
	{
		m[i] = new int[c+1];
		b[i] = new int[c+1];
	}
	//问题求解 
	BestV(c,w,p,n,m,b);
	x=new int[n+1];
	BestS(b,w,x,n,c);
	//输出最优值数组、决策数组 
	for(i=n;i>=2;i--)
	{
		cout<<"n="<<i<<":"<<endl;
		
		for(j=0;j<=c;j++)
			cout<<m[i][j]<<'\t';
		cout<<endl;
		
		for(j=0;j<=c;j++)
			cout<<b[i][j]<<'\t';
		cout<<endl;
	 } 
	 
	 //输出答案
	 cout<<"Best Value: "<<m[1][c]<<endl;
	 cout<<"Best Solution: ";
	 for(i=1;i<=n;i++)cout<<x[i]<<'\t';
	 cout<<endl;
	 for(i=1;i<=n;i++)
	 	if(x[i]) cout<<"物品"<<i<<"进包"<<endl;
	fclose(stdin);
	fclose(stdout);
	
	 return 0;
}

void BestV(int c, int *w, int *v, int n, int **m, int **b)
{
	int i,j;
	//初始化
	
	for(j=0;j<w[n];j++)m[n][j]=b[n][j]=0;
	for(j=w[n];j<=c;j++)m[n][j]=v[n],b[n][j]=1;
	
	for(i=n-1;i>=2;i--)
	{
		for(j=0;j<w[n];j++)m[i][j]=m[i+1][j], b[i][j]=0;
		
		for(j=w[i];j<=c;j++)
		{
			m[i][j]=m[i+1][j], b[i][j]=0;
			if(v[i]+m[i+1][j-w[i]]>m[i][j])
				m[i][j]=v[i]+m[i+1][j-w[i]], b[i][j]=1;
		}
	 } 
	 
	 m[1][c]=m[2][c], b[1][c]=0;
	 if(w[1]<=c)
	 {
	 	if(v[1]+m[2][c-w[i]]>m[1][c])
				m[1][c]=v[1]+m[2][c-w[1]], b[1][c]=1;
	 }
}

void BestS(int **b, int *w, int *x, int n,int c)
{
	int i,j;
	
	j=c;
	for(i=1;i<=n;i++)
	{
		x[i]=b[i][j];
		j-=x[i]*w[i];
	}
}
