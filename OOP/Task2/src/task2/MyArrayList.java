/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package task2;
//import java.util.ArrayList;
//import java.util.List;

/**
 *
 * @author Ayaaa
 */

public class MyArrayList {
    int[] ArrList = new int[32];
    int size;
    int capacity;
    int value;
    int count;
    int[] copy= new int[64];
    int arr;
    public void addElement(int element  )
    {  capacity=32;
       if (count>=capacity)
        {
        capacity=capacity*2;
        System.arraycopy(ArrList, 0, copy, 0, 32);
        copy[count]=element; 
        count++;
        arr=1;
        }
       else
       {
        if(capacity<32)
        {
       ArrList[count]=element; 
        count++;
        
        }
        else {
       
        copy[count]=element; 
        count++;
        }
       
       }
     
    }
    public int gitSize()
    {
     return count;   
    }
    public int gitCapacity()
    
    {
     return capacity;   
    }
    public int gitElement(int element)
    { 
      if (capacity>=32)
        {
        value=copy[element];
        }
       else
      { 
        value=ArrList[element];
      }
      return value;  
    }
}
