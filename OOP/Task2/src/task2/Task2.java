/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package task2;

/**
 *
 * @author Ayaaa
 */
public class Task2 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
       
        MyArrayList m=new MyArrayList();
        
        for(int i=0; i<40 ;i++)
        {
        m.addElement(i);
        
        }
        int x=m.gitSize();
        System.out.println(m.capacity);
        System.out.println(m.gitSize());
        System.out.println(m.gitElement(33));
 
    }
    
}
