/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package lp;

/**
 *
 * @author Gilma
 */

import java.util.HashSet;
import java.util.Set;

public class Node {
	
	private final int id;
	private int label;
	private Set<Integer> neighbors;
	
	public Node(int id, int label) {
		this.id=id;
		this.label=label;
		this.neighbors = new HashSet<>();
	}

	public int getId() {
		return id;
	}

	public int getLabel() {
		return label;
	}

	public void setLabel(int label) {
		this.label = label;
	}

	public Set<Integer> getNeighbors() {
		return neighbors;
	}

	public void setNeighbors(Set<Integer> neighbors) {
		this.neighbors = neighbors;
	}
	
	public void addNeighbor(int id) {
            boolean add;
            add = this.neighbors.add(id);
	}
	   
}
