#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    // 1. Parse command-line options.
    const char *mesh_file = "./star.mesh";
    int order = 1;
    bool static_cond = false;

    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    FiniteElementCollection *fec = new H1_FECollection(order, dim); 
	// класс H1 разбирали на паре
    FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

    // получаем индексы всех гран. узлов
	Array<int> ess_tdof_list;
    Array<int> ess_bdr;
    ess_bdr.SetSize(mesh->bdr_attributes.Max());
    ess_bdr = 1;
    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list); // главные гран. узлы

	LinearForm b(fespace);
    ConstantCoefficient one(1.0);
    b.AddDomainIntegrator(new DomainLFIntegrator(one));
    b.Assemble();

    GridFunction x(fespace); // вектор степеней свободы
    x = 0.0;

    BilinearForm a(fespace);
    a.AddDomainIntegrator(new DiffusionIntegrator(one));
    a.Assemble();

    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B); // исключает степени
													  // свободы дирихле

    OperatorJacobiSmoother M(a, ess_tdof_list);
    PCG(*A, M, B, X, 1, 1000, 1e-12, 0.0);
    
    /*UMFPackSolver umf_solver;
    umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_AMD;
    umf_solver.SetOperator(*A);
    umf_solver.Mult(B, X); //X = A^(-1)b*/

    a.RecoverFEMSolution(X, b, x);

    ofstream mesh_ofs("refined.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);
    ofstream sol_ofs("sol.gf");
    sol_ofs.precision(8);
    x.Save(sol_ofs);

    ofstream vtk_ofs("out_sol.vtk");
    vtk_ofs.precision(8);
    int ref = 0;
    mesh->PrintVTK(vtk_ofs, ref);
    x.SaveVTK(vtk_ofs, "u", ref);
    vtk_ofs.close();


    return 0;
}
