package task3;


public class task2 {

	

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		arraylist m=new arraylist();
        
        for(int i=0; i<40 ;i++)
        {
        m.addElement(i);
        
        }
        //int x=m.gitSize();
        System.out.println(m.capacity);
        System.out.println(m.gitSize());
        System.out.println(m.gitElement(33));


	}

}
