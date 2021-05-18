package task3;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class arraylistTest {

	@Test
	void test() {
		int cap, siz ,ele ;
     arraylist m=new arraylist();
        
        for(int i=0; i<30 ;i++)
        {
        m.addElement(i);
        
        }
        int x=m.gitSize();
        cap=m.capacity;
        siz=(m.gitSize());
        ele=(m.gitElement(22));
        
        assertTrue(cap == 32);
        assertTrue(siz == 30);
        assertTrue(ele == 22);
	}

}
